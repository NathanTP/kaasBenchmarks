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


def nshot(modelSpec, n, backend):
    backend.nShot(modelSpec, n, inline=False)


def runMlperf(modelSpec, backend):
    testing = False
    inline = False

    print("Starting MLPerf Benchmark: ")
    print("\tModel: ", modelSpec.name)
    print("\tBackend: ", backend.__name__)
    print("\tTesting: ", testing)
    print("\tInline: ", inline)

    backend.mlperfBench(modelSpec, testing=testing, inline=inline, useActors=False)


def main():
    parser = argparse.ArgumentParser("Inference benchmark driver")
    parser.add_argument("-m", "--model", help="Model to run")
    parser.add_argument("-b", "--backend", default='local', help="Which driver to use (local or ray)")
    parser.add_argument("-t", "--test", default="nshot", help="Which test to run (nshot or mlperf)")
    parser.add_argument("--testing", action="store_true", help="Run MLPerf in testing mode")
    parser.add_argument("--actors", action="store_true", help="Use actors for ray workloads")
    parser.add_argument("--inline", action="store_true", help="Inline pre and post processing with them model run (only meaningful for ray mode)")
    parser.add_argument("--numRun", default=1, type=int, help="Number of iterations to use in nshot mode")
    args = parser.parse_args()

    spec = util.getModelSpec(args.model)

    if args.backend == 'local':
        import localBench
        backend = localBench
    else:
        import rayBench
        backend = rayBench

    print(f"Starting {args.test} test")
    print("\t Model: ", args.model)
    print("\t Backend: ", args.backend)
    print("\t Testing: ", args.testing)
    print("\t Actors: ", args.actors)
    print("\t Inline: ", args.inline)

    if args.test == 'nshot':
        backend.nShot(spec, args.numRun, inline=args.inline, useActors=args.actors)
    elif args.test == 'mlperf':
        backend.mlperfBench(spec, testing=args.testing, inline=args.inline, useActors=args.actors)


main()
