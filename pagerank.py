#!/usr/bin/env python3
"""
Map: each rank computes contributions for its assigned source nodes.
Reduce: contributions are summed using MPI.Allreduce (native numeric reduction).
"""

from mpi4py import MPI
import argparse
import glob
import os
import re
import numpy as np

BETA = 0.9
ITERATIONS = 4

edge_re = re.compile(r"(\d+)\s+(\d+)")

def find_txt_files(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "*.txt")))

def scan_nodes(files):
    """Scan files and return sorted list of unique node ids (ints)."""
    nodes = set()
    for fp in files:
        with open(fp, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                s = s.replace("(", "").replace(")", "").replace(",", " ")
                m = edge_re.match(s)
                if not m:
                    continue
                src, dest = map(int, m.groups())
                nodes.add(src); nodes.add(dest)
    return sorted(nodes)

def build_local_adjacency(files, id2idx, rank, size):
    """
    Each rank builds adjacency only for source nodes assigned to this rank:
      src_idx % size == rank
    Returns:
      local_src_indices: list of source indices assigned to this rank (sorted)
      local_adj: dict src_idx -> numpy array of dest_indices
    """
    local_adj = {}  # src_idx -> list of dest indices (will convert to numpy arrays)
    for fp in files:
        with open(fp, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                s = s.replace("(", "").replace(")", "").replace(",", " ")
                m = edge_re.match(s)
                if not m:
                    continue
                src_id, dest_id = map(int, m.groups())
                # if either id not in id2idx (shouldn't happen), skip
                try:
                    src_idx = id2idx[src_id]
                    dest_idx = id2idx[dest_id]
                except KeyError:
                    continue
                if (src_idx % size) != rank:
                    continue
                if src_idx not in local_adj:
                    local_adj[src_idx] = []
                local_adj[src_idx].append(dest_idx)
    # convert lists to numpy arrays for efficient indexing later
    for k in list(local_adj.keys()):
        local_adj[k] = np.array(local_adj[k], dtype=np.int64)
    local_src_indices = sorted(local_adj.keys())
    return local_src_indices, local_adj

def parse_args():
    p = argparse.ArgumentParser(description="MPI PageRank (NumPy + MPI Allreduce)")
    p.add_argument("data_dir", help="Directory with input .txt edge files")
    p.add_argument("out_dir", help="Directory to write pagerank_iter_*.txt and top10.txt")
    p.add_argument("--iterations", type=int, default=ITERATIONS, help="Number of iterations")
    return p.parse_args()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    iterations = args.iterations

    os.makedirs(out_dir, exist_ok=True)

    # discover files
    if rank == 0:
        files = find_txt_files(data_dir)
        if not files:
            raise SystemExit(f"No .txt files found in {data_dir}")
        # build global node list
        node_list = scan_nodes(files)
        n = len(node_list)
    else:
        files = None
        node_list = None
        n = None

    # Broadcast file list and node_list length & values from root
    files = comm.bcast(files, root=0)
    n = comm.bcast(n, root=0)
    node_list = comm.bcast(node_list, root=0)

    if rank == 0:
        print(f"Found {len(files)} files, total nodes = {n}")
    comm.Barrier()

    # Build id -> index mapping locally on each rank (small)
    id2idx = {node_id: i for i, node_id in enumerate(node_list)}

    # Each rank builds adjacency for its assigned source nodes
    local_src_indices, local_adj = build_local_adjacency(files, id2idx, rank, size)
    print(f"Rank {rank}: built adjacency for {len(local_src_indices)} source nodes")

    # initialize ranks as numpy array (float64)
    ranks = np.full(n, 1.0 / n, dtype=np.float64)

    # temp arrays for contributions
    local_contrib = np.zeros(n, dtype=np.float64)
    global_contrib = np.zeros(n, dtype=np.float64)

    for it in range(iterations):
        # reset local_contrib
        local_contrib.fill(0.0)
        local_dangling = 0.0

        # compute local contributions
        for s in local_src_indices:
            out = local_adj.get(s, None)
            r = ranks[s]
            if out is None or out.size == 0:
                local_dangling += r
            else:
                # distribute r across out[]
                share = r / out.size
                # vectorized addition:
                # NOTE: out may have duplicates; adding multiple times is fine
                np.add.at(local_contrib, out, share)

        # sum contributions across ranks using Allreduce
        comm.Allreduce(local_contrib, global_contrib, op=MPI.SUM)

        # sum dangling mass across ranks
        total_dangling = comm.allreduce(local_dangling, op=MPI.SUM)

        # compute new ranks (everyone can compute using global_contrib & dangling)
        # v' = (1 - beta)/n + beta * (M v + dangling/n)
        new_ranks = (1.0 - BETA) / n + BETA * (global_contrib + total_dangling / n)

        # write intermediate file from rank 0 only
        if rank == 0:
            iter_file = os.path.join(out_dir, f"pagerank_iter_{it+1}.txt")
            with open(iter_file, "w") as fw:
                for idx, node_id in enumerate(node_list):
                    fw.write(f"{node_id}\t{new_ranks[idx]:.8f}\n")
            print(f"Iteration {it+1} complete — wrote {iter_file}")

        # prepare for next iteration: broadcast new ranks to all ranks
        # use Allreduce trick to broadcast array efficiently: use Bcast
        comm.Bcast(new_ranks, root=0)
        # update ranks variable
        ranks = new_ranks.copy()

    # after iterations, rank 0 computes top10 and writes file
    if rank == 0:
        top_idx = np.argsort(-ranks)[:10]  # descending
        top_file = os.path.join(out_dir, "top10.txt")
        with open(top_file, "w") as ft:
            for idx in top_idx:
                ft.write(f"{node_list[idx]}\t{ranks[idx]:.8f}\n")
        print(f"Top 10 written → {top_file}")

    comm.Barrier()
    if rank == 0:
        print("PageRank finished successfully.")

if __name__ == "__main__":
    main()
