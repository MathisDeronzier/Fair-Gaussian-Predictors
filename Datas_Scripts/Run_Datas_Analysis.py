#!/usr/bin/env python3
"""Run repeated fair GP experiments and save reproducible split results."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import hashlib
from importlib.metadata import version
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp

DEFAULT_RHOS = (-0.9, 0.0, 0.9, 1.0)
DEFAULT_TRAIN_CAP = 2000
DEFAULT_TEST_CAP = 2000
DEFAULT_GP_ITERS = 1000
DEFAULT_DOE_POINTS = 5
DEFAULT_PENALTY_ITERS = (20, 2400)
POSTERIOR_JITTER = 1e-8
BLUR = 0.05


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_cpu(item) for item in value]
    return value


def _shifted_cholesky(gram, shift):
    shifted = gram.clone()
    shifted.diagonal().add_(shift)
    return torch.linalg.cholesky(shifted)


def _exact_training_mean(gram, y, mean, noise):
    """Evaluate m + K(K + noise I)^(-1)(y-m) with one triangular solve pair."""
    factor = _shifted_cholesky(gram, noise)
    alpha = torch.cholesky_solve((y - mean).reshape(-1, 1), factor)
    return y.reshape(-1, 1) - noise * alpha


def _exact_training_precision(gram, noise):
    """Precision of the library's noise-added posterior at the training rows."""
    # With a = 2*noise + jitter and b = noise*(noise + jitter), the
    # posterior covariance is (a*K + b*I) @ (K + noise*I)^(-1).
    a = 2 * noise + POSTERIOR_JITTER
    shift = noise * (noise + POSTERIOR_JITTER) / a
    factor = _shifted_cholesky(gram, shift)
    precision = torch.cholesky_inverse(factor)
    precision.mul_((noise / a) ** 2)
    precision.diagonal().add_(1 / a)
    return precision


def _fit_gp(xs, y, rho, estimated, categorical, config, device):
    from Fair_GPs.Fair_GPs_classe import GP
    from Fair_GPs.utils import fair_regression

    seed = config.get("gp_seed", 42)
    torch.manual_seed(seed)
    gp = GP(xs, y, rho=rho, kernel_x="matern52", varnoise=0.1,
            inducing_m=config.get("inducing_m", 400),
            sparse_threshold=config.get("sparse_threshold", 7000),
            categorical_dims=categorical, categorical_corr=0.5, device=device)
    start_count = config.get("doe_points", DEFAULT_DOE_POINTS) + 1
    gp.Optimise_hyperparameters(
        iters=config.get("gp_iters", DEFAULT_GP_ITERS), lr=config.get("gp_lr", 0.005),
        fix_varnoise=False, fix_rho=not estimated, n_starts=start_count,
        design=config.get("design", "sobol"), seed=seed, verbose=config.get("verbose", False),
    )
    fitted = _cpu(gp.hyperparameters.as_dict())
    rho_fitted = gp.hyperparameters.rho
    with torch.no_grad():
        # Share the fitted prior Gram across noise calibration, posterior
        # statistics, and MAP interpolation. The sparse backend uses its
        # variational posterior for the first two steps.
        gram = gp.get_gram(xs)
        if gp.is_sparse:
            prediction = gp.get_cond_mean(xs)
        else:
            mean = gp.get_mean(xs)
            prediction = _exact_training_mean(gram, y, mean, fitted["varnoise"])
        prediction = prediction.reshape(-1)
    sensitive = xs[:, -1]
    cost = fair_regression(prediction[sensitive == 0], prediction[sensitive == 1])[2]
    heuristic_noise = float(cost + (prediction - y.reshape(-1)).square().mean())
    noise_floor = float(gp.likelihood.noise_covar.raw_noise_constraint.lower_bound)
    noise = max(heuristic_noise, float(np.nextafter(noise_floor, np.inf)))
    gp.set_hyperparameters(varnoise=noise)
    gp.model.train()
    gp.model.eval()
    if gp.hyperparameters.rho != rho_fitted:
        raise RuntimeError("Updating the noise variance changed the fitted rho.")
    noise = gp.hyperparameters.varnoise
    with torch.no_grad():
        if gp.is_sparse:
            posterior = gp._posterior_on(xs)
            mu = posterior.mean.reshape(-1, 1)
            precision = torch.cholesky_inverse(_shifted_cholesky(
                posterior.covariance_matrix, POSTERIOR_JITTER
            ))
        else:
            mu = _exact_training_mean(gram, y, mean, noise)
            precision = _exact_training_precision(gram, noise)
    if not torch.isfinite(mu).all() or not torch.isfinite(precision).all():
        raise RuntimeError("The GP posterior has non-finite values.")
    fit = {
        "hyperparameters_fitted": fitted, "hyperparameters_used": _cpu(gp.hyperparameters.as_dict()),
        "multistart_results": _cpu(gp.multistart_results), "fit_count": start_count,
        "mean_constant": float(gp.get_mean(xs[:1])[0]), "is_sparse": gp.is_sparse,
        "varnoise_heuristic": heuristic_noise, "varnoise_floor": noise_floor,
        "varnoise_floor_active": noise != heuristic_noise,
    }
    return gp, gram, mu, precision, fit


def _schedule(value, count, *, geometric=False):
    if isinstance(value, (tuple, list)):
        return np.geomspace(*value, count) if geometric else np.linspace(*value, count)
    return np.full(count, value)


def _optimize_path(method, mu, precision, group0_size, lambdas, config):
    from geomloss import SamplesLoss
    from Fair_GPs.Kernel_classe import MaternKernel, median_heuristic
    from Fair_GPs.Optim_Method import Optimize_fair_quad, mmd_loss

    blur = config.get("sinkhorn_blur", BLUR)
    group0, group1 = slice(None, group0_size), slice(group0_size, None)
    if method == "MMD":
        lengthscale = float(median_heuristic(mu))
        if not np.isfinite(lengthscale) or lengthscale <= 0:
            raise ValueError("The MMD bandwidth requires a positive median distance.")
        kernel = MaternKernel(nu=2.5, lengthscale=2 * lengthscale) + MaternKernel(nu=2.5, lengthscale=0.5 * lengthscale)
        discrepancy = {"nu": 2.5, "lengthscales": [2 * lengthscale, 0.5 * lengthscale]}
        divergence = lambda z: mmd_loss(z[group0], z[group1], kernel)
        budget = config.get("mmd_iters", DEFAULT_PENALTY_ITERS)
    else:
        kernel = None
        sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=BLUR, debias=True)
        discrepancy = {"p": 2, "blur": blur, "debias": True}
        divergence = lambda z: sinkhorn(z[group0], z[group1])
        budget = config.get("sinkhorn_iters", DEFAULT_PENALTY_ITERS)

    budgets = _schedule(budget, len(lambdas) - 1).round().astype(int)
    rates = _schedule(config.get("opt_lr", [0.005, 0.0005]), len(lambdas) - 1, geometric=True)
    solutions, logs = [mu], [None]
    divergences, quads, objectives, iterations = [float(divergence(mu))], [0.0], [0.0], [0]
    for weight, count, lr in zip(lambdas[1:], budgets, rates):
        z0, z1, distance, journal = Optimize_fair_quad(
            X0=solutions[-1][group0], X1=solutions[-1][group1],
            mu=mu, Gamma=precision, loss_type=method, kernel=kernel,
            blur=blur, p=2, lr=float(lr), n_iter=int(count), lambd=float(weight),
            tol=5e-4, autograd=True, verbose=False,
        )
        solution = torch.cat((z0, z1)).detach()
        if not torch.isfinite(solution).all():
            raise RuntimeError(f"{method}: optimization returned a non-finite solution.")
        difference = (solution - mu).flatten()
        quad = float(difference @ (precision @ difference))
        distance = float(distance)
        solutions.append(solution)
        logs.append(_cpu(journal))
        divergences.append(distance)
        quads.append(quad)
        objectives.append(quad + float(weight) * distance)
        iterations.append(len(journal["loss"]))
    return {"solutions": torch.cat(solutions, dim=1).T.contiguous(), "logs": logs,
            "divergences": divergences, "quad": quads, "objectives": objectives,
            "iterations": iterations, "discrepancy_settings": discrepancy,
            "requested_iterations": [0] + budgets.tolist(), "learning_rates": rates.tolist()}


@torch.no_grad()
def _predict(gp, gram, paths, data, batch_size, *, jitter=0.0, rtol=None, atol=0.0):
    # Apply one cached pseudoinverse to all lambda vectors from both penalties.
    widths = [len(path["solutions"]) for path in paths.values()]
    rhs = torch.cat([path["solutions"] for path in paths.values()]).T
    weights = gp.compute_map_coefficients(rhs, jitter=jitter, rtol=rtol, atol=atol)
    if not torch.isfinite(weights).all():
        raise RuntimeError("The MAP coefficients contain non-finite values.")
    for path, coefficients in zip(paths.values(), weights.T.split(widths)):
        path["coefficients"] = coefficients.detach().to(device="cpu", copy=True).contiguous()
        path["predictions"] = {}
        path["sensitive"] = {}
    for split in ("train", "test"):
        inputs = data[f"XS_{split}"].to(gp.device)
        outputs = {"observed": [], "cf0": [], "cf1": []}
        for start, batch in zip(range(0, len(inputs), batch_size), inputs.split(batch_size)):
            if split == "train":
                observed = (gram[start:start + len(batch)] @ weights).T
            else:
                observed = (gp.get_gram(batch, gp.X_train) @ weights).T
            if gp.model.covar_module.is_collapsed:
                pred0 = pred1 = observed
            else:
                counterfactual = batch.clone()
                counterfactual[:, -1] = 1 - batch[:, -1]
                flipped = (gp.get_gram(counterfactual, gp.X_train) @ weights).T
                at_zero = batch[:, -1] == 0
                pred0, pred1 = torch.where(at_zero, observed, flipped), torch.where(at_zero, flipped, observed)
            for key, values in (("observed", observed), ("cf0", pred0), ("cf1", pred1)):
                if not torch.isfinite(values).all():
                    raise RuntimeError("The MAP predictions contain non-finite values.")
                outputs[key].append(values.cpu())
        for kind, pieces in outputs.items():
            matrix = torch.cat(pieces, dim=1)
            for path, values in zip(paths.values(), matrix.split(widths)):
                # Store just this method's rows, even if split returned a view.
                path["predictions"].setdefault(split, {})[kind] = values.clone()
        for path in paths.values():
            path["sensitive"][split] = data[f"XS_{split}"][:, -1].contiguous()


def _ks(predictions, sensitive):
    from Fair_GPs.utils import KS

    return KS(predictions, sensitive)


def _metric_rows(identity, path, data):
    rows = []
    normalization = data["normalization"]
    train_sensitive = path["sensitive"]["train"]
    for index, solution in enumerate(path["solutions"]):
        solution_ks = _ks(solution, train_sensitive)
        solution_mse = float((solution - data["Y_train"]).square().mean())
        interpolation = float((path["predictions"]["train"]["observed"][index] - solution).abs().max())
        for split, predictions in path["predictions"].items():
            observed, cf0, cf1 = (predictions[key][index] for key in ("observed", "cf0", "cf1"))
            sensitive = path["sensitive"][split]
            delta = cf0 - cf1
            y, y_original = data[f"Y_{split}"], data[f"Y_{split}_original"]
            original = observed * normalization["y_scale"] + normalization["y_mean"]
            rows.append({**identity, "split": split, "lambda_index": index,
                         "lambda": float(path["lambdas"][index]),
                         "lambda_relative": float(path["lambda_relative"][index]), "n": len(y),
                         "MSE": float((observed - y).square().mean()),
                         "MSE_original": float((original - y_original).square().mean()),
                         "KS": _ks(observed, sensitive), "disp": float(delta.square().mean()),
                         "cf_max_abs": float(delta.abs().max()), "cf_mean_abs": float(delta.abs().mean()),
                         "ks_counterfactual": float(ks_2samp(cf0.numpy(), cf1.numpy()).statistic),
                         "solution_KS": solution_ks, "solution_MSE": solution_mse,
                         "divergence": path["divergences"][index], "quad": path["quad"][index],
                         "objective": path["objectives"][index], "iterations": path["iterations"][index],
                         "interpolation_max_abs": interpolation})
    return rows


def run_split(dataset, seed, config, data_dir, output):
    """Run a complete dataset/split job in an empty caller-owned staging directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"The split staging directory is not empty: {output}")
    from Datas_Scripts.Data_Preprocessing import load_dataset
    from Fair_GPs.Data_Processing import _prepare_split

    source = load_dataset(dataset, data_dir)
    # Import the library's modules before setting the worker's default device.
    from Fair_GPs import Fair_GPs_classe, Optim_Method, Kernel_classe  # noqa: F401

    device = torch.device(config.get("device", "cpu"))
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(config.get("threads", 2))
    data = _prepare_split(source, seed, config.get("train_cap", DEFAULT_TRAIN_CAP),
                          config.get("test_cap", DEFAULT_TEST_CAP))
    torch.set_default_device(device)
    data["dataset"] = dataset
    xs, y = data["XS_train"].to(device), data["Y_train"].to(device)
    group0_size = int((xs[:, -1] == 0).sum())
    save_artifacts = config.get("save_artifacts", True)
    split_data = data if save_artifacts else {
        key: value for key, value in data.items() if not key.startswith(("XS_", "Y_"))
    }
    torch.save(_cpu(split_data), output / "split.pt")
    relative = torch.cat((torch.zeros(1), torch.exp(torch.linspace(
        config.get("lambda_min_exp", -1.0), config.get("lambda_max_exp", 9.0),
        config.get("lambda_count", 8),
    ))))
    if not torch.isfinite(relative).all() or not (relative[1:] > 0).all():
        raise ValueError("Lambda exponents must produce finite positive factors.")
    specs = [(f"rho_{rho:g}", rho, False) for rho in config.get("fixed_rhos", DEFAULT_RHOS)]
    specs.append(("rho_star", config.get("rho_initial", 0.0), True))
    methods = config.get("methods", ("MMD", "Sinkhorn"))
    jitter = config.get("prediction_jitter", 0.0)
    rtol = config.get("prediction_rtol", None)
    atol = config.get("prediction_atol", 0.0)
    inversion_settings = {"method": "symmetric_pseudoinverse", "jitter": jitter,
                          "rtol": rtol, "atol": atol}
    rows, model_rows = [], []
    for key, rho, estimated in specs:
        started = time.perf_counter()
        print(f"{dataset} split={seed} model={key} train={len(y)} test={len(data['Y_test'])}", flush=True)
        gp, gram, mu, precision, fit = _fit_gp(xs, y, rho, estimated, data["categorical_dims"], config, device)
        identity = {"dataset": dataset, "run": seed, "model_key": key,
                    "rho": gp.hyperparameters.rho, "rho_estimated": estimated}
        lambdas = torch.linalg.norm(precision, ord="fro") * relative
        if not torch.isfinite(lambdas).all():
            raise ValueError("The scaled lambda weights exceed the numerical range.")
        paths = {method: _optimize_path(method, mu, precision, group0_size, lambdas, config) for method in methods}
        _predict(gp, gram, paths, data, config.get("prediction_batch_size", 512),
                 jitter=jitter, rtol=rtol, atol=atol)
        inversion_info = dict(gp.map_pinv_info)
        model_dir = output / "models" / key
        if save_artifacts:
            model_dir.mkdir(parents=True)
            posterior = {**identity, **fit, "mu": mu.flatten(), "gamma_inv": precision,
                         "kernel_train": gram, "prediction_jitter": jitter,
                         "prediction_rtol": rtol, "prediction_atol": atol,
                         "prediction_inversion": inversion_settings,
                         "prediction_inversion_info": inversion_info,
                         "model_state_dict": gp.model.state_dict(), "likelihood_state_dict": gp.likelihood.state_dict()}
            if config.get("save_kernel_inverse", False):
                posterior["kernel_regularized_inv"] = gp.get_pinv_gram(jitter=jitter, rtol=rtol, atol=atol)
            torch.save(_cpu(posterior), model_dir / "posterior.pt")
            del posterior
            (model_dir / "paths").mkdir()
        for method, path in paths.items():
            path.update(identity, method=method, lambdas=lambdas.cpu(), lambda_relative=relative.cpu(),
                        solutions=path["solutions"].cpu(), prediction_inversion=inversion_settings,
                        prediction_inversion_info=inversion_info)
            rows.extend(_metric_rows({**identity, "method": method}, path, data))
            if save_artifacts:
                torch.save(_cpu(path), model_dir / "paths" / f"{method}.pt")
        hp = fit["hyperparameters_used"]
        selected = next((result for result in fit["multistart_results"] if result["selected"]), None)
        model_rows.append({**identity, **hp, "lengthscale": json.dumps(hp["lengthscale"]),
                           "varnoise_fitted": fit["hyperparameters_fitted"]["varnoise"],
                           "varnoise_heuristic": fit["varnoise_heuristic"],
                           "varnoise_floor_active": fit["varnoise_floor_active"],
                           "mean_constant": fit["mean_constant"], "is_sparse": fit["is_sparse"],
                           "fit_count": fit["fit_count"], "selected_fit": selected["start_index"] if selected else 0,
                           "training_objective": selected["objective"] if selected else "",
                           "training_loss": selected["loss"] if selected else None,
                           "seconds": time.perf_counter() - started})
        del gp, mu, precision, gram, paths, path
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(["method", "split", "rho", "rho_estimated", "lambda_index"])
    if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
        raise RuntimeError("The experiment produced non-finite metrics.")
    frame.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(model_rows).sort_values(["rho", "rho_estimated"]).to_csv(output / "models.csv", index=False)
    return {"dataset": dataset, "run": seed, "n_train": len(y),
            "n_test": len(data["Y_test"]), "metric_rows": len(frame), "model_count": len(model_rows)}


# Server orchestration

SOURCE_FILES = (
    "Fair_GPs/Fair_GPs_classe.py", "Fair_GPs/Data_Processing.py",
    "Datas_Scripts/Data_Preprocessing.py", "Fair_GPs/Optim_Method.py",
    "Fair_GPs/utils.py", "Fair_GPs/Kernel_classe.py",
)
FIXED_RHOS = (-1.0, -0.9, 0.0, 0.9, 1.0)
SCHEMA_VERSION = 1


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return result


def parse_cap(value):
    return None if value.lower() == "none" else positive_int(value)


def parse_args():
    # Resolve imports before building dataset choices, including external copies
    # launched with --repo-root and workers started with the spawn context.
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--repo-root", type=Path)
    requested, _ = bootstrap.parse_known_args()
    root = project_root(requested.repo_root)
    sys.path.insert(0, str(root))
    from Datas_Scripts.Data_Preprocessing import DATA_FILES

    parser = argparse.ArgumentParser(description="Run repeated fair GP experiments on a multicore Linux server.")
    parser.add_argument("--repo-root", type=Path, help="Repository root if the script is stored elsewhere.")
    parser.add_argument("--data-dir", type=Path, help="Dataset folder, default REPO_ROOT/Datas_Scripts/data.")
    parser.add_argument("--output", type=Path, required=True, help="One result folder per experiment.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATA_FILES), default=list(DATA_FILES))
    parser.add_argument("--splits", type=positive_int, help="Number of consecutive split seeds, default 30 (2 with --smoke).")
    parser.add_argument("--first-seed", type=int, default=0, help="First split seed, default 0.")
    parser.add_argument("--train-cap", type=parse_cap, default=argparse.SUPPRESS,
                        help=f"Maximum training rows per split, default {DEFAULT_TRAIN_CAP} (100 with --smoke), or none.")
    parser.add_argument("--test-cap", type=parse_cap, default=argparse.SUPPRESS,
                        help=f"Maximum test rows per split, default {DEFAULT_TEST_CAP} (200 with --smoke), or none.")
    parser.add_argument("--workers", type=positive_int, default=1, help="Independent split jobs running concurrently.")
    parser.add_argument("--threads", type=positive_int, default=1, help="CPU threads per worker.")
    parser.add_argument("--gp-iters", type=positive_int, default=DEFAULT_GP_ITERS,
                        help=f"Maximum GP iterations per start, default {DEFAULT_GP_ITERS}.")
    parser.add_argument("--doe-points", type=positive_int, default=DEFAULT_DOE_POINTS,
                        help=f"Generated starts per model, default {DEFAULT_DOE_POINTS}, in addition to its baseline.")
    parser.add_argument("--design", choices=["sobol", "lhs"], default="sobol")
    parser.add_argument("--gp-seed", type=int, default=42, help="Base seed for GP fitting and designs.")
    parser.add_argument("--lambda-count", type=positive_int, default=8, help="Number of positive penalty weights.")
    parser.add_argument("--lambda-min-exp", type=float, default=-1.0)
    parser.add_argument("--lambda-max-exp", type=float, default=9.0)
    parser.add_argument("--mmd-iters", nargs=2, type=positive_int, default=list(DEFAULT_PENALTY_ITERS),
                        metavar=("MIN", "MAX"), help=f"MMD iteration budget endpoints, default {DEFAULT_PENALTY_ITERS[0]} {DEFAULT_PENALTY_ITERS[1]}.")
    parser.add_argument("--sinkhorn-iters", nargs=2, type=positive_int, default=list(DEFAULT_PENALTY_ITERS),
                        metavar=("MIN", "MAX"), help=f"Sinkhorn iteration budget endpoints, default {DEFAULT_PENALTY_ITERS[0]} {DEFAULT_PENALTY_ITERS[1]}.")
    parser.add_argument("--prediction-jitter", type=float, default=0.0,
                        help="Explicit diagonal shift before the MAP pseudoinverse, default 0.")
    parser.add_argument("--prediction-rtol", type=float, default=None,
                        help="Relative eigenvalue cutoff for the MAP pseudoinverse, default automatic (N times machine epsilon).")
    parser.add_argument("--prediction-atol", type=float, default=0.0,
                        help="Absolute eigenvalue cutoff for the MAP pseudoinverse, default 0.")
    artifact_options = parser.add_mutually_exclusive_group()
    artifact_options.add_argument("--metrics-only", action="store_true", help="Keep CSVs and split records. Skip dense model tensors.")
    artifact_options.add_argument("--save-kernel-inverse", action="store_true", help="Also save the pseudoinverse used for MAP predictions.")
    parser.add_argument("--resume", action="store_true", help="Continue a matching experiment from completed split folders.")
    parser.add_argument("--smoke", action="store_true", help="Use one design point, 3 GP iterations, and 2 penalty iterations per weight.")
    parser.add_argument("--plan", action="store_true", help="Print the effective settings and job counts, then exit.")
    return parser.parse_args()


def project_root(requested):
    roots = [requested.resolve()] if requested else [
        Path(__file__).resolve().parent,
        *Path(__file__).resolve().parent.parents,
        Path.cwd(),
    ]
    for root in roots:
        if (root / "Fair_GPs" / "Fair_GPs_classe.py").is_file():
            return root
    raise FileNotFoundError("Use --repo-root to locate the updated Fair-Gaussian-Predictors repository.")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def repository_revision(root):
    git = shutil.which("git")
    if not git:
        return None
    result = subprocess.run([git, "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def effective_config(args):
    return {
        "smoke": args.smoke,
        "device": "cpu",
        "train_cap": getattr(args, "train_cap", 100 if args.smoke else DEFAULT_TRAIN_CAP),
        "test_cap": getattr(args, "test_cap", 200 if args.smoke else DEFAULT_TEST_CAP),
        "gp_iters": 3 if args.smoke else args.gp_iters,
        "doe_points": 1 if args.smoke else args.doe_points,
        "design": args.design,
        "gp_seed": args.gp_seed,
        "gp_lr": 5e-3,
        "lambda_count": args.lambda_count,
        "lambda_min_exp": args.lambda_min_exp,
        "lambda_max_exp": args.lambda_max_exp,
        "methods": ["MMD", "Sinkhorn"],
        "fixed_rhos": list(FIXED_RHOS),
        "prediction_jitter": args.prediction_jitter,
        "prediction_rtol": args.prediction_rtol,
        "prediction_atol": args.prediction_atol,
        "prediction_batch_size": 512,
        "mmd_iters": [2, 2] if args.smoke else args.mmd_iters,
        "sinkhorn_iters": [2, 2] if args.smoke else args.sinkhorn_iters,
        "opt_lr": [5e-3, 5e-4],
        "sinkhorn_blur": 0.05,
        "save_kernel_inverse": args.save_kernel_inverse,
        "save_artifacts": not args.metrics_only,
        "inducing_m": 400,
        "sparse_threshold": 7000,
    }


def experiment_manifest(args, root, data_dir, config, seeds):
    from Datas_Scripts.Data_Preprocessing import DATA_FILES

    paths = sorted({name for dataset in args.datasets for name in DATA_FILES[dataset]})
    missing = [name for name in paths if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files in {data_dir}: {', '.join(missing)}. "
            "Download them with python Datas_Scripts/download_datasets.py "
            f"--datasets {' '.join(args.datasets)} --data-dir \"{data_dir}\". "
            "See Datas_Scripts/README.md for the public sources."
        )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "datasets": args.datasets,
        "split_seeds": seeds,
        "test_fraction": 0.30,
        "config": config,
        "data_sha256": {name: file_hash(data_dir / name) for name in paths},
        "source_sha256": {
            "runner": file_hash(Path(__file__)),
            **{name: file_hash(root / name) for name in SOURCE_FILES},
        },
        "versions": {
            name: version(name)
            for name in ("torch", "gpytorch", "linear-operator", "POT", "geomloss",
                         "numpy", "pandas", "scipy", "scikit-learn")
        },
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        **identity,
        "experiment_id": hashlib.sha256(serialized.encode()).hexdigest(),
        "repository_revision": repository_revision(root),
        "data_dir": str(data_dir),
        "command": sys.argv,
    }


def expected_rows(config):
    return len(config["methods"]) * 2 * (len(config["fixed_rhos"]) + 1) * (config["lambda_count"] + 1)


def split_directory(output, dataset, seed):
    return output / dataset / f"split_{seed:03d}"


def is_complete(folder, dataset, seed, experiment_id, row_count):
    if not folder.exists():
        return False
    marker_file = folder / "complete.json"
    if not marker_file.is_file():
        raise ValueError(f"Incomplete result folder {folder}. A complete split needs complete.json.")
    marker = json.loads(marker_file.read_text(encoding="utf-8"))
    if (marker.get("experiment_id") != experiment_id or marker.get("dataset") != dataset
            or marker.get("run") != seed or marker.get("metric_rows") != row_count
            or marker.get("metrics_sha256") != file_hash(folder / "metrics.csv")):
        raise ValueError(f"Saved result does not match this experiment or is damaged: {folder}")
    return True


def initialize_worker(root, threads):
    # The initializer also runs in serial mode, before any fitting starts.
    sys.path.insert(0, str(root))
    from threadpoolctl import threadpool_limits
    # Hold the controller for the lifetime of the worker.
    global _thread_limits
    _thread_limits = threadpool_limits(limits=threads)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cpu")


def execute_split(dataset, seed, config, data_dir, output, experiment_id):
    target = split_directory(output, dataset, seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".split_{seed:03d}_", dir=target.parent))
    started = time.monotonic()
    try:
        run_split(dataset, seed, config, data_dir, staging)
        metrics_file = staging / "metrics.csv"
        with metrics_file.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        keys = {(row["method"], row["split"], row["model_key"], int(row["lambda_index"])) for row in rows}
        row_count = expected_rows(config)
        if len(rows) != row_count or len(keys) != row_count:
            raise RuntimeError(f"{dataset} seed {seed} produced an incomplete or duplicate metric grid.")
        if any(row["dataset"] != dataset or int(row["run"]) != seed for row in rows):
            raise RuntimeError(f"{dataset} seed {seed} produced mismatched metric labels.")
        seconds = time.monotonic() - started
        atomic_json(staging / "complete.json", {
            "experiment_id": experiment_id, "dataset": dataset, "run": seed,
            "metric_rows": row_count, "metrics_sha256": file_hash(metrics_file),
            "elapsed_seconds": seconds, "cpu_threads": config["threads"],
        })
        # Publish a complete directory in one rename on the output filesystem.
        try:
            os.rename(staging, target)
        except OSError:
            if not is_complete(target, dataset, seed, experiment_id, row_count):
                raise
        return seconds
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def combine_metrics(output, tasks, experiment_id, row_count):
    path = output / "metrics.csv"
    temporary = path.with_suffix(".csv.tmp")
    count = 0
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = None
        for dataset, seed in tasks:
            folder = split_directory(output, dataset, seed)
            if not is_complete(folder, dataset, seed, experiment_id, row_count):
                continue
            with (folder / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if writer is None:
                    writer = csv.DictWriter(target, reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
                    count += 1
    if count:
        os.replace(temporary, path)
    else:
        temporary.unlink()
    return count


def run_experiment(args, root, data_dir, output, manifest, tasks):
    path = output / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if not args.resume:
            raise ValueError("This experiment already exists. Use --resume to continue it.")
        if previous["experiment_id"] != manifest["experiment_id"]:
            raise ValueError("The data, code, dependencies, or scientific settings changed. Use a new output folder.")
    elif args.resume:
        raise ValueError("--resume requires an existing manifest.json.")
    else:
        atomic_json(path, manifest)

    experiment_id = manifest["experiment_id"]
    config = {**manifest["config"], "threads": args.threads}
    row_count = expected_rows(config)
    pending = [(d, s) for d, s in tasks
               if not is_complete(split_directory(output, d, s), d, s, experiment_id, row_count)]
    print(f"{len(tasks) - len(pending)}/{len(tasks)} splits already complete. "
          f"Workers: {args.workers}, CPU threads per worker: {args.threads}.", flush=True)
    failures = []
    completed = len(tasks) - len(pending)

    def report(dataset, seed, result):
        nonlocal completed
        try:
            seconds = result()
        except Exception as error:
            failures.append({"dataset": dataset, "run": seed, "error": str(error),
                             "traceback": traceback.format_exc()})
            print(f"Failed: {dataset} seed={seed}. {error}", file=sys.stderr, flush=True)
        else:
            completed += 1
            print(f"Complete {completed}/{len(tasks)}: {dataset} seed={seed} in {seconds:.1f}s.", flush=True)

    interrupted = False
    try:
        if pending and args.workers == 1:
            initialize_worker(root, args.threads)
            for dataset, seed in pending:
                report(dataset, seed, lambda: execute_split(dataset, seed, config, data_dir, output, experiment_id))
        elif pending:
            # Spawn keeps Torch's thread pools out of the process-fork path.
            executor = ProcessPoolExecutor(
                max_workers=args.workers, mp_context=mp.get_context("spawn"),
                initializer=initialize_worker, initargs=(root, args.threads),
            )
            try:
                futures = {executor.submit(execute_split, d, s, config, data_dir, output, experiment_id): (d, s)
                           for d, s in pending}
                for future in as_completed(futures):
                    dataset, seed = futures[future]
                    report(dataset, seed, future.result)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        count = combine_metrics(output, tasks, experiment_id, row_count)
        atomic_json(output / "status.json", {
            "planned_splits": len(tasks), "completed_splits": count // row_count,
            "metric_rows": count, "failures": failures, "interrupted": interrupted,
        })
        print(f"Saved {count} metric rows under {output}.", flush=True)
    if interrupted:
        return 130
    return 0 if count == len(tasks) * row_count else 1


def main():
    args = parse_args()
    root = project_root(args.repo_root)
    data_dir = (args.data_dir or root / "Datas_Scripts" / "data").resolve()
    output = args.output.resolve()
    config = effective_config(args)
    split_count = args.splits if args.splits is not None else (2 if args.smoke else 30)
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Dataset names must be unique.")
    if not 0 <= args.first_seed <= 2**32 - split_count or not 0 <= args.gp_seed < 2**32:
        raise ValueError("Seeds must be nonnegative and fit within the 32-bit split seed range.")
    if not (np.isfinite(args.lambda_min_exp) and np.isfinite(args.lambda_max_exp)
            and args.lambda_min_exp < args.lambda_max_exp):
        raise ValueError("Lambda exponents must be finite and increasing.")
    prediction_tolerances = [args.prediction_jitter, args.prediction_atol]
    if args.prediction_rtol is not None:
        prediction_tolerances.append(args.prediction_rtol)
    if any(not np.isfinite(value) or value < 0 for value in prediction_tolerances):
        raise ValueError("Prediction jitter and pseudoinverse tolerances must be finite and nonnegative.")
    if any(bounds[0] > bounds[1] for bounds in (args.mmd_iters, args.sinkhorn_iters)):
        raise ValueError("Penalty iteration bounds must be in increasing order.")

    seeds = list(range(args.first_seed, args.first_seed + split_count))
    tasks = [(d, s) for d in args.datasets for s in seeds]
    fits = len(tasks) * (len(FIXED_RHOS) + 1) * (config["doe_points"] + 1)
    print(f"Plan: {len(tasks)} dataset/split jobs, {fits} GP starts, "
          f"{config['gp_iters']} GP iterations per start, "
          f"{len(tasks) * expected_rows(config)} metric rows.", flush=True)
    if args.plan:
        print(json.dumps(config, indent=2, allow_nan=False))
        return 0

    # Set limits before spawning. The initializer also limits libraries already loaded.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    manifest = experiment_manifest(args, root, data_dir, config, seeds)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another runner is writing to {output}.") from error
        return run_experiment(args, root, data_dir, output, manifest, tasks)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
