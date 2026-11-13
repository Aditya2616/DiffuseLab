"""Keep every number in README.md identical to the results files.

    python scripts/check_numbers.py --check    # exit 1 if the README is out of sync (a fast test runs this)
    python scripts/check_numbers.py --write    # regenerate the README blocks and inline values

The README holds two kinds of generated text:

* blocks between ``<!-- BEGIN:name -->`` and ``<!-- END:name -->`` (the result tables), and
* inline values written as ``<!--v:key-->text<!--/v-->`` (numbers quoted in prose).

Both are rendered from docs/data/*.json (training.json, convergence.json, benchmark.json, benchmark_cpu.json,
cpu_timing.json, index.json), the training presets in src/diffuselab/train.py and the image files under docs/.
The UI reads the same JSON files, so the README, the UI and the data cannot drift apart without this check failing.
Only the standard library is needed.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
README = ROOT / "README.md"

NAMES = {"ddpm": "DDPM", "ddim": "DDIM", "dpmpp": "DPM-Solver++", "euler": "Euler (Karras)"}


def load(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def preset_tiny() -> dict:
    """PRESETS["tiny"] read from train.py without importing torch."""
    tree = ast.parse((ROOT / "src" / "diffuselab" / "train.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "PRESETS":
            presets = {k.value: {kw.arg: ast.literal_eval(kw.value) for kw in v.keywords} for k, v in zip(node.value.keys, node.value.values, strict=True)}
            return presets["tiny"]
    raise SystemExit("PRESETS not found in train.py")


def image_megabytes() -> float:
    """Total size of the binary images under docs/ (text files are left out: their size depends on line endings)."""
    total = sum(p.stat().st_size for p in (ROOT / "docs").rglob("*") if p.suffix in {".png", ".webp"})
    return total / 1e6


def n(v, d=0) -> str:
    return f"{v:,.{d}f}"


def fid_verdict(fid: dict) -> str:
    """A plain verdict from fid.json. A metric favours a schedule only if the gap exceeds three times the larger seed
    standard deviation; otherwise that metric is tied. A schedule wins a sampler only if both FID and KID favour it;
    otherwise the two are not separated, and the sentence says, from the numbers, what each metric did. The spread is
    over sampling seeds only (each schedule was trained once)."""
    marks: dict[str, dict[str, int]] = {}  # sampler -> metric -> 1: cosine lower (better), -1: linear lower, 0: tied
    cells: dict[str, tuple[dict, dict]] = {}
    for key in fid["results"]["linear"]:
        a, b = fid["results"]["linear"][key], fid["results"]["cosine"][key]
        cells[key] = (a, b)
        marks[key] = {}
        for m in ("fid", "kid"):
            d = b[f"{m}_mean"] - a[f"{m}_mean"]
            sd = max(a[f"{m}_std"] or 0, b[f"{m}_std"] or 0)
            marks[key][m] = 0 if abs(d) <= 3 * sd else (1 if d < 0 else -1)
    wins = {k: ("cosine" if v == {"fid": 1, "kid": 1} else "linear" if v == {"fid": -1, "kid": -1} else "neither") for k, v in marks.items()}

    def name(key: str) -> str:
        s, st = key.split("-")
        return f"{NAMES[s]} at {st} steps"

    def who(v: int) -> str:
        return "cosine" if v == 1 else "linear"

    def clause(key: str) -> str:
        """What FID and KID each did under one sampler, with the two means where a metric is tied."""
        mk, (a, b) = marks[key], cells[key]
        tied = {m: f"{m.upper()} is tied ({1000 * a['kid_mean'] if m == 'kid' else a['fid_mean']:.2f} vs {1000 * b['kid_mean'] if m == 'kid' else b['fid_mean']:.2f})" for m in ("fid", "kid")}
        if mk["fid"] == 0 and mk["kid"] == 0:
            return f"under {name(key)} FID and KID are both tied (FID {a['fid_mean']:.2f} vs {b['fid_mean']:.2f}, KID x 1000 {1000 * a['kid_mean']:.2f} vs {1000 * b['kid_mean']:.2f})"
        if mk["fid"] == 0 or mk["kid"] == 0:
            m_tied, m_win = ("fid", "kid") if mk["fid"] == 0 else ("kid", "fid")
            return f"under {name(key)} {tied[m_tied]} while {m_win.upper()} favours {who(mk[m_win])}"
        return f"under {name(key)} FID favours {who(mk['fid'])} and KID favours {who(mk['kid'])}"

    won = {k: v for k, v in wins.items() if v != "neither"}
    rest = [k for k, v in wins.items() if v == "neither"]
    if not won:
        return "I cannot separate the two schedules under any sampler: " + "; ".join(clause(k) for k in rest) + "."
    out = "; ".join(f"the {v} schedule is better under {name(k)} (lower FID and KID by more than three times the seed spread)" for k, v in won.items())
    if rest:
        out += "; " + "; ".join(clause(k) for k in rest) + ", so I cannot separate them there"
    winners = set(won.values())
    tail = f" The {'cosine' if winners == {'linear'} else 'linear'} schedule does not win under any sampler measured." if len(winners) == 1 else " Neither schedule wins everywhere."
    return out[0].upper() + out[1:] + "." + tail


def fid_headline(fid: dict) -> str:
    """One sentence for the top of the README: how many of the measured samplers separate the two schedules."""
    wins = []
    for key in fid["results"]["linear"]:
        a, b = fid["results"]["linear"][key], fid["results"]["cosine"][key]
        mk = []
        for m in ("fid", "kid"):
            d = b[f"{m}_mean"] - a[f"{m}_mean"]
            sd = max(a[f"{m}_std"] or 0, b[f"{m}_std"] or 0)
            mk.append(0 if abs(d) <= 3 * sd else (1 if d < 0 else -1))
        wins.append("cosine" if mk == [1, 1] else "linear" if mk == [-1, -1] else "neither")
    total = len(wins)
    return (f"Measured result: of the {total} samplers I scored with FID and KID, the linear schedule is clearly better under {wins.count('linear')}, "
            f"the cosine schedule under {wins.count('cosine')}, and the other {wins.count('neither')} are not separated (details in Results).")


def fid_sampler_note(fid: dict) -> str:
    best = {kind: min(c, key=lambda k: c[k]["fid_mean"]) for kind, c in fid["results"].items()}
    if set(best.values()) == {"ddim-50"}:
        return "For both schedules, DDIM with 50 steps has the lowest FID of the samplers measured, the usual price of fewer network calls at 20."
    return "The best sampler by FID differs between the schedules (see the table)."


def values() -> dict[str, str]:
    t, conv, bg, bc, cpu, idx = (load(x) for x in ("training.json", "convergence.json", "benchmark.json", "benchmark_cpu.json", "cpu_timing.json", "index.json"))
    first, fid = load("first_sample.json"), load("fid.json")
    lin, cos = t["runs"]["linear"], t["runs"]["cosine"]
    for key in ("params", "steps", "epochs_run", "batch", "lr", "ema_decay", "seed", "train_images", "gpu", "torch", "device"):
        assert lin[key] == cos[key], f"the two runs differ in {key}: they must share their settings"
    tiny = preset_tiny()
    assert cpu["steps"] == (lin["train_images"] // cpu["batch"]) * cpu["epochs"], "cpu_timing.json does not match the tiny preset"
    per_epoch = lin["steps"] // lin["epochs_run"]
    rows = {(r["scheduler"], r["steps"]): r["seconds_median"] for r in bg["rows"]}
    rows_cpu = {(r["scheduler"], r["steps"]): r["seconds_median"] for r in bc["rows"]}
    conv_by = {r["steps"]: r for r in conv["rows"]}
    below = [t["nn_check"][k]["generated_below_heldout_p1"] for k in t["nn_check"]]
    nn_claim = (
        "No generated sample is closer to a training image than the closest 1% of held-out real images are."
        if all(b == 0 for b in below)
        else "Some generated samples are closer to a training image than the closest 1% of held-out real images (see the table in `training.json`)."
    )
    held_med = t["nn_check"]["linear"]["heldout_to_train_median"]
    below_keys = [k for k in t["samplers"]["linear"] if any(t["samplers"][kind][k]["generated_to_train_median"] < held_med for kind in ("linear", "cosine"))]
    nn_below = "; ".join(
        f"{NAMES[k.split('-')[0]]}, {k.split('-')[1]} steps (linear {t['samplers']['linear'][k]['generated_to_train_median']:.2f}, cosine {t['samplers']['cosine'][k]['generated_to_train_median']:.2f})"
        for k in below_keys
    ) or "none"
    by_model: dict[str, set[int]] = {}
    for r in idx:
        by_model.setdefault(r["model"], set()).add(r["seed"])
    samplers = {r["scheduler"] for r in idx}
    lr = f"{lin['lr']:.0e}".replace("e-0", "e-")
    fv = {
        "fid_n_gen": n(fid["meta"]["n_generated"]),
        "fid_n_real": n(fid["meta"]["n_real"]),
        "fid_seeds": str(len(fid["meta"]["sampling_seeds"])),
        "fid_kid_subset": n(fid["meta"]["kid_subset_size"]),
        "fid_tf_version": fid["meta"]["torch_fidelity"],
        "fid_floor": f"{fid['floor']['fid_mean']:.2f}",
        "fid_floor_sd": f"{fid['floor']['fid_std']:.2f}",
        "kid_floor": f"{1000 * fid['floor']['kid_mean']:.2f}",
        "fid_verdict": fid_verdict(fid),
        "fid_headline": fid_headline(fid),
        "fid_sampler_note": fid_sampler_note(fid),
    }
    for kind, cells in fid["results"].items():
        for key, c in cells.items():
            tag = f"{kind}_{key.replace('-', '_')}"
            fv[f"fid_{tag}"] = f"{c['fid_mean']:.2f}"
            fv[f"fidsd_{tag}"] = "n/a" if c["fid_std"] is None else f"{c['fid_std']:.2f}"
            fv[f"kid_{tag}"] = f"{1000 * c['kid_mean']:.2f}"
    return fv | {
        "params": n(lin["params"]),
        "params_m": f"{lin['params'] / 1e6:.1f}M",
        "steps": n(lin["steps"]),
        "epochs": str(lin["epochs_run"]),
        "steps_per_epoch": n(per_epoch),
        "batch": str(lin["batch"]),
        "lr": lr,
        "ema": str(lin["ema_decay"]),
        "seed": str(lin["seed"]),
        "train_images": n(lin["train_images"]),
        "holdout": n(t["holdout"]),
        "gpu": lin["gpu"],
        "torch": lin["torch"],
        "lin_s": n(lin["seconds"]),
        "lin_min": f"{lin['seconds'] / 60:.1f}",
        "lin_loss": f"{lin['final_epoch_loss']:.4f}",
        "cos_s": n(cos["seconds"]),
        "cos_min": f"{cos['seconds'] / 60:.1f}",
        "cos_loss": f"{cos['final_epoch_loss']:.4f}",
        "full_minutes": f"{round(lin['seconds'] / 60)} to {round(cos['seconds'] / 60)}",
        "nn_n": str(t["nn_check"]["linear"]["n"]),
        "heldout_median": f"{t['nn_check']['linear']['heldout_to_train_median']:.2f}",
        "nn_claim": nn_claim,
        "nn_below": nn_below,
        "lin_nn": f"{t['nn_check']['linear']['generated_to_train_median']:.2f}",
        "cos_nn": f"{t['nn_check']['cosine']['generated_to_train_median']:.2f}",
        "conv_seeds": str(conv["n_seeds"]),
        "gpu_repeats": str(len(bg["rows"][0]["seconds_all"])),
        "cpu_repeats": str(len(bc["rows"][0]["seconds_all"])),
        "dpm20_rmse": f"{conv_by[20]['dpmpp']:.3f}",
        "ddim50_rmse": f"{conv_by[50]['ddim']:.3f}",
        "dpm20_cost_pct": f"{100 * rows[('dpmpp', 20)] / rows[('ddim', 50)]:.0f}%",
        "bench_torch": bg["torch"],
        "bench_card": bg["gpu"],
        "cpu_bench_torch": bc["torch"],
        "tiny_epochs": str(tiny["epochs"]),
        "tiny_widths": "/".join(str(w) for w in tiny["widths"]),
        "cpu_steps": n(cpu["steps"]),
        "cpu_epochs": str(cpu["epochs"]),
        "cpu_batch": str(cpu["batch"]),
        "cpu_params": n(cpu["params"]),
        "cpu_params_m": f"{cpu['params'] / 1e6:.1f}M",
        "cpu_seconds": n(cpu["seconds"]),
        "cpu_minutes": f"{cpu['seconds'] / 60:.1f}",
        "cpu_loss": f"{cpu['final_epoch_loss']:.4f}",
        "cpu_threads": str(cpu["threads"]),
        "cpu_torch": cpu["torch"],
        "cpu_size": str(tiny["size"]),
        "cpu_load_note": cpu["load_note"],
        "first_load_note": first["load_note"],
        "first_s": f"{first['seconds_median']:.1f}",
        "first_repeats": str(first["repeats"]),
        "first_threads": str(first["threads"]),
        "n_runs": str(len(idx)),
        "n_samplers": str(len(samplers)),
        "cifar_seeds": str(len(by_model["cifar10"])),
        "fashion_seeds": str(len(by_model["fashion-linear"])),
        "image_mb": f"{image_megabytes():.1f}",
        "ddpm_cuda_s": f"{rows[('ddpm', 1000)]:.1f}",
        "ddpm_cpu_s": f"{rows_cpu[('ddpm', 1000)]:.0f}",
    }


def blocks() -> dict[str, str]:
    t, conv, bg, bc = (load(x) for x in ("training.json", "convergence.json", "benchmark.json", "benchmark_cpu.json"))
    lin, cos = t["runs"]["linear"], t["runs"]["cosine"]
    out = {}
    out["training"] = "\n".join(
        [
            "| schedule | steps | wall time | final-epoch loss | wall time measured on |",
            "|---|---|---|---|---|",
            f"| linear | {n(lin['steps'])} | {n(lin['seconds'])} s ({lin['seconds'] / 60:.1f} min) | {lin['final_epoch_loss']:.4f} | {lin['gpu']}, torch {lin['torch']}, bf16 autocast, CUDA graph; GPU mostly to itself, CPU shared with other jobs |",
            f"| cosine | {n(cos['steps'])} | {n(cos['seconds'])} s ({cos['seconds'] / 60:.1f} min) | {cos['final_epoch_loss']:.4f} | {cos['gpu']}, torch {cos['torch']}, bf16 autocast, CUDA graph; run alone on the GPU, CPU shared with other jobs |",
        ]
    )
    names = list(t["samplers"]["linear"])
    nn = ["| sampler | linear: median NN distance | cosine: median NN distance |", "|---|---|---|"]
    for key in names:
        s, st = key.split("-")
        label = f"{NAMES[s]}, {st} steps"
        nn.append(f"| {label} | {t['samplers']['linear'][key]['generated_to_train_median']:.2f} | {t['samplers']['cosine'][key]['generated_to_train_median']:.2f} |")
    out["nn"] = "\n".join(nn)
    nc = t["nn_check"]
    out["nn_closest"] = "\n".join(
        ["| schedule (DDIM, 50 steps) | closest generated sample | closest held-out image | generated samples below the held-out 1% quantile |", "|---|---|---|---|"]
        + [f"| {kind} | {nc[kind]['generated_to_train_min']:.2f} | {nc[kind]['heldout_to_train_min']:.2f} | {round(nc[kind]['generated_below_heldout_p1'] * nc[kind]['n'])} of {nc[kind]['n']} |" for kind in ("linear", "cosine")]
    )
    cv = ["| steps | DDIM | DPM-Solver++ | Euler (Karras) |", "|---|---|---|---|"]
    cv += [f"| {r['steps']} | {r['ddim']:.4f} | {r['dpmpp']:.4f} | {r['euler']:.4f} |" for r in conv["rows"]]
    out["convergence"] = "\n".join(cv)
    fid = load("fid.json")
    ft = ["| schedule | sampler | seeds | FID (mean ± sd) | KID × 1000 (mean ± sd) |", "|---|---|---|---|---|"]
    fl = fid["floor"]
    ft.append(f"| real training images (floor) | none | {len(fl['runs'])} | {fl['fid_mean']:.2f} ± {fl['fid_std']:.2f} | {1000 * fl['kid_mean']:.2f} ± {1000 * fl['kid_std']:.2f} |")
    for key in fid["results"]["linear"]:
        for kind in ("linear", "cosine"):
            c = fid["results"][kind][key]
            s, st_ = key.split("-")
            sd = "n/a (1 seed)" if c["fid_std"] is None else f"{c['fid_std']:.2f}"
            ksd = "n/a" if c["kid_std"] is None else f"{1000 * c['kid_std']:.2f}"
            ft.append(f"| {kind} | {NAMES[s]}, {st_} steps | {len(c['runs'])} | {c['fid_mean']:.2f} ± {sd} | {1000 * c['kid_mean']:.2f} ± {ksd} |")
    out["fid"] = "\n".join(ft)
    cpu = {(r["scheduler"], r["steps"]): r["seconds_median"] for r in bc["rows"]}
    bt = ["| sampler | network calls | GPU (s) | CPU (s) |", "|---|---|---|---|"]
    for r in bg["rows"]:
        bt.append(f"| {NAMES[r['scheduler']]} | {r['steps']} | {r['seconds_median']:.2f} | {cpu[(r['scheduler'], r['steps'])]:.1f} |")
    out["benchmark"] = "\n".join(bt)
    return out


def render(text: str) -> str:
    v, b = values(), blocks()
    for name, body in b.items():
        text, k = re.subn(rf"(<!-- BEGIN:{name} -->\n).*?(\n<!-- END:{name} -->)", lambda m, body=body: m.group(1) + body + m.group(2), text, flags=re.S)
        if k != 1:
            raise SystemExit(f"README needs exactly one BEGIN/END marker pair for '{name}' (found {k})")

    def sub(m: re.Match) -> str:
        if m.group(1) not in v:
            raise SystemExit(f"README uses an unknown inline value '{m.group(1)}'")
        return f"<!--v:{m.group(1)}-->{v[m.group(1)]}<!--/v-->"

    return re.sub(r"<!--v:(\w+)-->.*?<!--/v-->", sub, text)


def main(argv: list[str]) -> int:
    mode = "--write" if "--write" in argv else "--check"
    cur = README.read_text(encoding="utf-8").replace("\r\n", "\n")
    new = render(cur)
    if mode == "--write":
        README.write_text(new, encoding="utf-8", newline="\n")
        print("README.md regenerated" if new != cur else "README.md already in sync")
        return 0
    if new == cur:
        print("README numbers match the results files")
        return 0
    import difflib

    sys.stdout.writelines(difflib.unified_diff(cur.splitlines(True), new.splitlines(True), "README.md", "from results files", n=0))
    print("\nREADME.md is out of sync with docs/data. Run: python scripts/check_numbers.py --write")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
