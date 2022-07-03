#!/usr/bin/env python
import util
import argparse
import datetime
from kaas.pool import policies


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
    parser.add_argument("-b", "--backend",
                        default='local', choices=['local', 'ray', 'client'],
                        help="Which driver to use (local or ray)")
    parser.add_argument("-e", "--experiment",
                        default="nshot", choices=['nshot', 'mlperf', 'server', 'throughput', 'deepProf'],
                        help="Which test to run")
    parser.add_argument("--testing", action="store_true", help="Run MLPerf in testing mode")
    parser.add_argument("-p", "--policy",
                        choices=['exclusive', 'balance', 'static'], default='balance',
                        help="Scheduling policy to use for actor and KaaS mode.")
    parser.add_argument("--force-cold", action="store_true", dest='forceCold',
                        help="Force cold starts if possible (this is only valid in some configurations)")
    parser.add_argument("--inline", action="store_true",
                        help="Inline pre and post processing with them model run (only meaningful for ray mode)")
    parser.add_argument("--scale", type=float,
                        help="Rate at which to submit requests in mlperf mode (as a fraction of peak throughput). If not provided, mlperf is run in FindPeakPerformance mode.")
    parser.add_argument("--runTime", type=float,
                        help="Target runtime for experiment in seconds (only valid for throughput and mlperf tests).")
    parser.add_argument("--numRun", default=1, type=int,
                        help="Number of iterations to use in nshot mode")
    parser.add_argument("--numClient", default=1, type=int,
                        help="Expected number of clients in server mode. This is used to implement a barrier.")
    parser.add_argument("--fractional", default=None, choices=['mem', 'sm'],
                        help="In server mode, assign fractional GPUs to clients based on the specified resource (memory or SM)")
    parser.add_argument("--mig", default=False, action="store_true", help="Emulate MIG (only valid for the static policy and with --fractional set)")
    args = parser.parse_args()

    print(f"Starting {args.experiment} experiment")
    print("\t Model: ", args.model)
    print("\t Backend: ", args.backend)
    print("\t Testing: ", args.testing)
    print("\t Runner Policy: ", args.policy)
    print("\t Force Cold: ", args.forceCold)
    print("\t Inline: ", args.inline)
    print("\t Fractional: ", args.fractional)

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

    if args.fractional is not None and args.policy != 'static':
        raise ValueError("'fractional' can only be used with the static policy")

    if args.policy == 'balance':
        policy = policies.BALANCE
    elif args.policy == 'exclusive':
        policy = policies.EXCLUSIVE
    elif args.policy == 'static':
        policy = policies.STATIC
    else:
        raise ValueError("Unsupported policy: ", args.policy)

    benchConfig = {
        "time": datetime.datetime.today().strftime("%y-%m-%d:%d:%H:%M:%S"),
        "gitHash": util.currentGitHash(),
        "name": args.name,
        "model": args.model,
        "experiment": args.experiment,
        "backend": args.backend,
        "testing": args.testing,
        "policy": policy,
        "forceCold": args.forceCold,
        "inline": args.inline,
        "scale": args.scale,
        "runTime": args.runTime,
        "numRun": args.numRun,
        "numClient": args.numClient,
        "fractional": args.fractional,
        "mig": args.mig
    }

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
    elif args.experiment == 'deepProf':
        # Deep prof isn't as automated as other tests, you'll have to mess with
        # it manually for most things
        if args.backend != 'local':
            raise ValueError("Deep Profile only available in local mode")
        spec = util.getModelSpec(args.model)
        benchConfig['model_type'] = spec.modelType
        backend.deepProfile(spec, benchConfig)
    else:
        raise ValueError("Unrecognized test: ", args.test)


main()
