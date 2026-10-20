"""Run the full study over several independently simulated worlds and save results.

Usage:  python run_all.py --seeds 5
Outputs: results/results.pkl, results/summary.json, results/*.csv, figures/*.pdf|png
"""
import argparse, json, multiprocessing, os, pickle, sys, time, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # nan-SDs when --seeds 1
import numpy as np
from rcd import experiment

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if getattr(sys, "frozen", False):  # PyInstaller exe: write results/ and figures/ next to the exe
        os.chdir(os.path.dirname(sys.executable))
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--first", type=int, default=2026)
    ap.add_argument("--no-browser", action="store_true", help="don't open report.html when done")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    out = []
    t0 = time.time()
    for k in range(args.seeds):
        out.append(experiment.run(args.first + k))
        pickle.dump(out, open("results/results.pkl", "wb"))
    print(f"all seeds done in {time.time()-t0:.0f}s")
    from rcd import report
    report.make(out)
    print(f"outputs written to {os.path.abspath('results')} and {os.path.abspath('figures')}")
    from rcd import webpage
    page = webpage.make()
    print(f"results web page: {page}")
    if not args.no_browser:
        import webbrowser
        webbrowser.open("file:///" + page.replace(os.sep, "/"))
    if getattr(sys, "frozen", False) and len(sys.argv) == 1:  # launched by double-click
        input("Press Enter to close...")
