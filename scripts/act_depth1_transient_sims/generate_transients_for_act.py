import argparse
import json
import sys

sys.path.append("../../../sotrplib/")
from sotrplib.sims import sim_utils as su
from sotrplib.sims.sim_sources import generate_transients


def main():
    parser = argparse.ArgumentParser(
        description="Generate transient sources and save to a database."
    )
    parser.add_argument(
        "-n", type=int, required=True, help="Number of transient sources to generate."
    )
    parser.add_argument(
        "--ra-lims", type=float, nargs=2, default=(0, 360), help="RA limits (min, max)."
    )
    parser.add_argument(
        "--dec-lims",
        type=float,
        nargs=2,
        default=(-60, 20),
        help="Dec limits (min, max).",
    )
    parser.add_argument(
        "--flux-lims",
        type=float,
        nargs=2,
        default=(0.01, 10),
        help="Flux limits in Jy (min, max).",
    )
    parser.add_argument(
        "--unix-time-lims",
        type=float,
        nargs=2,
        default=(1.4944e9, 1.6568e9),
        help="Unix time limits for ACT (min, max).",
    )
    parser.add_argument(
        "--flare-width-lims",
        type=float,
        nargs=2,
        default=(60, 86400),
        help="Flare width limits in seconds (min, max).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="act_depth1_transient_sim_db.db",
        help="Path to the database file.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="act_depth1_transient_sim_config.json",
        help="Path to save the configuration file.",
    )
    args = parser.parse_args()

    positions = su.generate_random_positions(
        args.n, ra_lims=args.ra_lims, dec_lims=args.dec_lims
    )
    fluxes = su.generate_random_flare_amplitudes(
        args.n, min_amplitude=args.flux_lims[0], max_amplitude=args.flux_lims[1]
    )

    peak_times = su.generate_random_flare_times(
        args.n, start_time=1.4944e9, end_time=1.6568e9
    )

    flare_widths = su.generate_random_flare_widths(
        args.n, min_width=args.flare_width_lims[0], max_width=args.flare_width_lims[1]
    )
    flare_morphs = ["Gaussian"] * args.n
    beam_params = [{}] * args.n
    transients = generate_transients(
        positions=positions,
        peak_amplitudes=fluxes,
        peak_times=peak_times,
        flare_widths=flare_widths,
        flare_morphs=flare_morphs,
        beam_params=beam_params,
    )
    su.save_transients_to_db(transients, args.db_path)

    # Save configuration to file
    config = {
        "n": args.n,
        "ra_lims": args.ra_lims,
        "dec_lims": args.dec_lims,
        "flux_lims": args.flux_lims,
        "flare_width_lims": args.flare_width_lims,
    }
    with open(args.config_path, "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    main()
