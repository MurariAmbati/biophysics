"""
Command-line interface for DNA Knot Visualizer.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

from dna_knot.generators import TorusKnotGenerator, PrimeKnotGenerator, RandomPolygonGenerator
from dna_knot.invariants import compute_invariants_summary
from dna_knot.simplification import minimize_energy
from dna_knot.visualization import plot_knot_3d, save_knot_3d, export_planar_diagram_svg
from dna_knot.io import save_session, load_session, export_obj, export_json
from dna_knot.io.export import export_diagram_json


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DNA Knot Visualizer - Generate, analyze, and visualize knotted curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate a knot')
    gen_parser.add_argument('--type', required=True, choices=['torus', 'prime', 'random'],
                           help='Knot generator type')
    gen_parser.add_argument('--p', type=int, help='Torus knot parameter p')
    gen_parser.add_argument('--q', type=int, help='Torus knot parameter q')
    gen_parser.add_argument('--knot', help='Prime knot name (unknot, trefoil, figure_eight, etc.)')
    gen_parser.add_argument('--mode', choices=['uniform', 'equilateral', 'self_avoiding'],
                           default='uniform', help='Random polygon mode')
    gen_parser.add_argument('--R', type=float, default=2.0, help='Major radius (torus)')
    gen_parser.add_argument('--r', type=float, default=0.5, help='Minor radius (torus)')
    gen_parser.add_argument('--N', type=int, default=512, help='Number of vertices')
    gen_parser.add_argument('--seed', type=int, help='Random seed')
    gen_parser.add_argument('--out', required=True, help='Output session file (.json)')
    
    # Compute command
    comp_parser = subparsers.add_parser('compute', help='Compute invariants')
    comp_parser.add_argument('--session', required=True, help='Input session file')
    comp_parser.add_argument('--invariants', action='store_true', help='Compute all invariants')
    comp_parser.add_argument('--out', help='Output file for invariants (JSON)')
    
    # Simplify command
    simp_parser = subparsers.add_parser('simplify', help='Simplify knot diagram')
    simp_parser.add_argument('--session', required=True, help='Input session file')
    simp_parser.add_argument('--method', choices=['energy'], default='energy',
                            help='Simplification method')
    simp_parser.add_argument('--steps', type=int, default=2000, help='Max iterations')
    simp_parser.add_argument('--out', required=True, help='Output session file')
    
    # Export command
    exp_parser = subparsers.add_parser('export', help='Export visualization or geometry')
    exp_parser.add_argument('--session', required=True, help='Input session file')
    exp_parser.add_argument('--format', required=True, choices=['svg', 'obj', 'json', 'png'],
                           help='Export format')
    exp_parser.add_argument('--out', required=True, help='Output file')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Show interactive visualization')
    viz_parser.add_argument('--session', required=True, help='Input session file')
    viz_parser.add_argument('--show-vertices', action='store_true', help='Show vertices')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Dispatch commands
    if args.command == 'generate':
        cmd_generate(args)
    elif args.command == 'compute':
        cmd_compute(args)
    elif args.command == 'simplify':
        cmd_simplify(args)
    elif args.command == 'export':
        cmd_export(args)
    elif args.command == 'visualize':
        cmd_visualize(args)


def cmd_generate(args):
    """Generate a knot."""
    print(f"Generating {args.type} knot...")
    
    if args.type == 'torus':
        if not args.p or not args.q:
            print("Error: --p and --q required for torus knots")
            sys.exit(1)
        
        gen = TorusKnotGenerator(
            p=args.p,
            q=args.q,
            R=args.R,
            r=args.r,
            N=args.N,
            seed=args.seed
        )
    
    elif args.type == 'prime':
        if not args.knot:
            print("Error: --knot required for prime knots")
            sys.exit(1)
        
        gen = PrimeKnotGenerator(
            knot_type=args.knot,
            N=args.N,
            seed=args.seed
        )
    
    elif args.type == 'random':
        gen = RandomPolygonGenerator(
            N=args.N,
            mode=args.mode,
            seed=args.seed
        )
    
    knot = gen.generate()
    print(f"Generated knot with {knot.n_vertices} vertices")
    
    # Save session
    save_session(knot, args.out)


def cmd_compute(args):
    """Compute invariants."""
    print(f"Loading session from {args.session}...")
    knot, _, _ = load_session(args.session)
    
    if args.invariants:
        print("Computing invariants...")
        diagram = knot.project()
        invariants = compute_invariants_summary(diagram)
        
        print("\n=== Invariants ===")
        for key, value in invariants.items():
            print(f"  {key}: {value}")
        
        # Save invariants if output specified
        if args.out:
            import json
            with open(args.out, 'w') as f:
                json.dump(invariants, f, indent=2)
            print(f"\nSaved invariants to {args.out}")


def cmd_simplify(args):
    """Simplify knot."""
    print(f"Loading session from {args.session}...")
    knot, _, _ = load_session(args.session)
    
    print(f"Simplifying using {args.method} method...")
    
    if args.method == 'energy':
        simplified_knot = minimize_energy(
            knot,
            max_iters=args.steps,
            verbose=True,
            check_topology=True
        )
    
    print(f"Saving simplified knot to {args.out}...")
    save_session(simplified_knot, args.out)


def cmd_export(args):
    """Export knot or diagram."""
    print(f"Loading session from {args.session}...")
    knot, invariants, _ = load_session(args.session)
    
    print(f"Exporting to {args.format}...")
    
    if args.format == 'svg':
        diagram = knot.project()
        export_planar_diagram_svg(diagram, args.out)
    
    elif args.format == 'obj':
        export_obj(knot, args.out)
    
    elif args.format == 'json':
        export_json(knot, args.out)
    
    elif args.format == 'png':
        save_knot_3d(knot, args.out)


def cmd_visualize(args):
    """Visualize knot interactively."""
    print(f"Loading session from {args.session}...")
    knot, _, _ = load_session(args.session)
    
    import matplotlib.pyplot as plt
    from dna_knot.visualization.plot3d import plot_knot_with_projection
    
    print("Showing visualization...")
    plot_knot_with_projection(knot)
    plt.show()


if __name__ == '__main__':
    main()
