#!/usr/bin/env python
import util
import argparse
import datetime


def sanityCheck(backend):
    """Basic check to make sure nothing is obviously broken. This is meant to
    be manually fiddled with to spot check stuff. It will run the superres
    model and write the output to test.png, it should be the superres output (a
    small cat next to a big cat in a figure)."""
    spec = util.getModelSpec("superResKaas")
    res = backend.nShot(spec, 1)

    with open("test.png", "wb") as f:
        f.write(res[0][0])

    print("Sanity check didn't crash!")
    print("Output available at ./test.png")


def main():
    parser = argparse.ArgumentParser("Inference benchmark driver")
    parser.add_argument("-n", "--name", default="test", help="Name to use internally and when saving results")
    parser.add_argument("-m", "--model", help="Model to run")
    parser.add_argument("-b", "--backend", default='local', choices=['local', 'ray', 'client'], help="Which driver to use (local or ray)")
    parser.add_argument("-e", "--experiment", default="nshot", choices=['nshot', 'mlperf', 'server', 'throughput'], help="Which test to run")
    parser.add_argument("--testing", action="store_true", help="Run MLPerf in testing mode")
    parser.add_argument("-p", "--policy", choices=['rr', 'exclusive', 'affinity', 'balance', 'hedge'], default=None, help="Scheduling policy to use for actor and KaaS mode.")
    parser.add_argument("--no_cache", action="store_true", help="Don't cache models on workers")
    parser.add_argument("--inline", action="store_true", help="Inline pre and post processing with them model run (only meaningful for ray mode)")
    parser.add_argument("--scale", type=float, help="Rate at which to submit requests in mlperf mode (as a fraction of peak throughput). If not provided, mlperf is run in FindPeakPerformance mode.")
    parser.add_argument("--numRun", default=1, type=int, help="Number of iterations to use in nshot mode")
    parser.add_argument("--numClient", default=1, type=int, help="Expected number of clients in server mode. This is used to implement a barrier.")
    args = parser.parse_args()

    if args.backend == 'local':
        import localBench
        backend = localBench
    elif args.backend == 'ray':
        import rayBench
        backend = rayBench
    elif args.backend == 'client':
        import client
        backend = client
    else:
        raise ValueError("Unrecognized backend: " + args.backend)

    benchConfig = {
        "time": datetime.datetime.today().strftime("%y-%m-%d:%d:%H:%M:%S"),
        "gitHash": util.currentGitHash(),
        "name": args.name,
        "model": args.model,
        "experiment": args.experiment,
        "backend": args.backend,
        "testing": args.testing,
        "policy": args.policy,
        "cache": not args.no_cache,
        "inline": args.inline,
        "scale": args.scale,
        "numRun": args.numRun,
        "numClient": args.numClient
    }

    print(f"Starting {args.experiment} experiment")
    print("\t Model: ", args.model)
    print("\t Backend: ", args.backend)
    print("\t Testing: ", args.testing)
    print("\t Runner Policy: ", args.policy)
    print("\t Cache Models: ", not args.no_cache)
    print("\t Inline: ", args.inline)

    if args.experiment == 'nshot':
        spec = util.getModelSpec(args.model)
        benchConfig['model_type'] = spec.modelType
        backend.nShot(spec, args.numRun, benchConfig)
    elif args.experiment == 'mlperf':
        spec = util.getModelSpec(args.model)
        benchConfig['model_type'] = spec.modelType
        backend.mlperfBench(spec, benchConfig)
    elif args.experiment == 'server':
        backend.serveRequests(benchConfig)
    elif args.experiment == 'throughput':
        spec = util.getModelSpec(args.model)
        benchConfig['model_type'] = spec.modelType
        backend.throughput(spec, benchConfig)
    else:
        raise ValueError("Unrecognized test: ", args.test)


main()
