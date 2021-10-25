#!/usr/bin/env python
import os
import sys

import click
import torch
import torch.nn.functional as F
import numpy as np

from model.model import Model as Model


def load_model(chk_path):
    model = Model().eval()
    state_dict = torch.load(chk_path)
    model.load_state_dict(state_dict)
    return model

def xyz2pdb(seq, CA):
    one_to_three = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR', 'X': "UNK"}
    line = "ATOM%7i  CA  %s A%4i    %8.3f%8.3f%8.3f  1.00  0.00           C"
    ret = []
    for i in range(CA.shape[0]):
        ret.append(line % (i + 1, one_to_three[seq[i]], i + 1, CA[i][0], CA[i][1], CA[i][2]))
    ret.append("TER")
    return ret


@click.command()
@click.option(
    "-i",
    "--feat",
    help="CopulaNet output",
    required=True,
    type=click.Path(exists=True),
)
@click.option("-f", "--fasta", help="Fasta format sequence", required=True)
@click.option("-o", "--output", help="Output pdb", required=True)
@click.option("--cuda", is_flag=True)
def main(fasta, feat, output, cuda):
    model = load_model(os.path.join(os.path.dirname(__file__), "weights/weights.pth"))
    seq = open(fasta).readlines()[1].strip()
    feat = np.load(feat)
    pwt_feat = np.concatenate(
        [feat["cbcb"], feat["omega"], feat["theta"], feat["phi"]], axis=-1
    )
    assert pwt_feat.shape[0] == len(seq)
    AMINO = "ACDEFGHIKLMNPQRSTVWY"
    seq_feat = [AMINO.index(_) if _ in AMINO else 20 for _ in seq]

    pwt_feat = torch.tensor(pwt_feat).float()[None]
    seq_feat = torch.tensor(seq_feat).long()[None]

    if cuda:
        pwt_feat = pwt_feat.cuda()
        seq_feat = seq_feat.cuda()
        model = model.cuda()
    with torch.no_grad():
        ret = model({"seq": seq_feat, "pwt_feat": pwt_feat})
    with open(output, "w") as fp:
        fp.write("\n".join(xyz2pdb(seq, ret[0][0])))


if __name__ == "__main__":
    main()
