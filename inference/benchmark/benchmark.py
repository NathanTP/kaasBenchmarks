#!/usr/bin/env python
import util
import argparse


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
    parser.add_argument("-m", "--model",
                        choices=['testModel', 'bert', 'resnet50', 'superRes', 'cGEMM', 'jacobi'],
                        help="Model to run.")
    parser.add_argument("-t", "--modelType", default='native',
                        choices=['kaas', 'native'], help="Which model type to use")
    parser.add_argument("-b", "--backend",
                        default='local', choices=['local', 'ray', 'client'],
                        help="Which driver to use (local or ray)")
    parser.add_argument("-e", "--experiment",
                        default="nshot", choices=['nshot', 'mlperf', 'server', 'throughput', 'deepProf'],
                        help="Which test to run")
    parser.add_argument("-p", "--policy",
                        choices=['exclusive', 'balance', 'static', 'affinity'], default='balance',
                        help="Scheduling policy to use for actor and KaaS mode.")
    parser.add_argument("--force-cold", action="store_true", dest='forceCold',
                        help="Force cold starts if possible (this is only valid in some configurations)")
    parser.add_argument("--inline", action="store_true",
                        help="Inline pre and post processing with them model run (only meaningful for ray mode)")
    parser.add_argument("-s", "--scale", type=float, help="For mlperf modes, what scale to run each client at. If omitted, tests will try to find peak performance. For nshot, this is the number of iterations.")
    parser.add_argument("--runTime", type=float,
                        help="Target runtime for experiment in seconds (only valid for throughput and mlperf tests).")
    parser.add_argument("--numClient", default=1, type=int,
                        help="Expected number of clients in server mode. This is used to implement a barrier.")
    parser.add_argument("--fractional", default=None, choices=['mem', 'sm'],
                        help="In server mode, assign fractional GPUs to clients based on the specified resource (memory or SM)")
    parser.add_argument("--name", default="test", help="Name to use internally and when saving results")

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

    if args.fractional is not None and args.policy != 'static':
        raise ValueError("'fractional' can only be used with the static policy")

    benchConfig = util.argsToConfig(args)

    print(f"Starting {args.experiment} experiment")
    print(f"\t Model: {benchConfig['model']} ({benchConfig['modelType']})")
    print("\t Experiment Key: ", benchConfig['expKey'])
    print("\t Backend: ", benchConfig['backend'])
    print("\t Runner Policy: ", benchConfig['policy'])
    print("\t Force Cold: ", benchConfig['forceCold'])
    print("\t Inline: ", benchConfig['inline'])
    print("\t Fractional: ", benchConfig['fractional'])

    if args.experiment != "server":
        spec = util.getModelSpec(args.model, args.modelType)

    if args.experiment == 'nshot':
        benchConfig['model_type'] = spec.modelType
        backend.nShot(spec, int(args.scale), benchConfig)
    elif args.experiment == 'mlperf':
        if args.backend != 'client':
            print("WARNING: mlperf mode directly in benchmark.py is deprecated and may not work correctly. Use experiment.py --nCopy=1 instead.")
        benchConfig['model_type'] = spec.modelType
        backend.mlperfBench(spec, benchConfig)
    elif args.experiment == 'server':
        backend.serveRequests(benchConfig)
    elif args.experiment == 'throughput':
        if args.backend != 'client':
            print("WARNING: throughput mode directly in benchmark.py is deprecated and may not work correctly. Use experiment.py --nCopy=1 instead.")
        benchConfig['model_type'] = spec.modelType
        backend.throughput(spec, benchConfig)
    elif args.experiment == 'deepProf':
        # Deep prof isn't as automated as other tests, you'll have to mess with
        # it manually for most things
        if args.backend != 'local':
            raise ValueError("Deep Profile only available in local mode")
        benchConfig['model_type'] = spec.modelType
        backend.deepProfile(spec, benchConfig)
    else:
        raise ValueError("Unrecognized test: ", args.test)


main()
