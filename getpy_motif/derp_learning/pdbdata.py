import numpy as np
import pyrosetta


all_score_types = {
    "fa_atr": pyrosetta.rosetta.core.scoring.ScoreType.fa_atr,
    "fa_rep": pyrosetta.rosetta.core.scoring.ScoreType.fa_rep,
    "fa_sol": pyrosetta.rosetta.core.scoring.ScoreType.fa_sol,
    "fa_intra_atr_xover4": pyrosetta.rosetta.core.scoring.ScoreType.fa_intra_atr_xover4,
    "fa_intra_rep_xover4": pyrosetta.rosetta.core.scoring.ScoreType.fa_intra_rep_xover4,
    "fa_intra_sol_xover4": pyrosetta.rosetta.core.scoring.ScoreType.fa_intra_sol_xover4,
    "lk_ball": pyrosetta.rosetta.core.scoring.ScoreType.lk_ball,
    "lk_ball_iso": pyrosetta.rosetta.core.scoring.ScoreType.lk_ball_iso,
    "lk_ball_bridge": pyrosetta.rosetta.core.scoring.ScoreType.lk_ball_bridge,
    "lk_ball_bridge_uncpl": pyrosetta.rosetta.core.scoring.ScoreType.lk_ball_bridge_uncpl,
    "fa_elec": pyrosetta.rosetta.core.scoring.ScoreType.fa_elec,
    "fa_intra_elec": pyrosetta.rosetta.core.scoring.ScoreType.fa_intra_elec,
    "pro_close": pyrosetta.rosetta.core.scoring.ScoreType.pro_close,
    "hbond_sr_bb": pyrosetta.rosetta.core.scoring.ScoreType.hbond_sr_bb,
    "hbond_lr_bb": pyrosetta.rosetta.core.scoring.ScoreType.hbond_lr_bb,
    "hbond_bb_sc": pyrosetta.rosetta.core.scoring.ScoreType.hbond_bb_sc,
    "hb_sc": pyrosetta.rosetta.core.scoring.ScoreType.hbond_sc,
    "dslf_fa13": pyrosetta.rosetta.core.scoring.ScoreType.dslf_fa13,
    "rama_prepro": pyrosetta.rosetta.core.scoring.ScoreType.rama_prepro,
    "omega": pyrosetta.rosetta.core.scoring.ScoreType.omega,
    "p_aa_pp": pyrosetta.rosetta.core.scoring.ScoreType.p_aa_pp,
    "fa_dun_rot": pyrosetta.rosetta.core.scoring.ScoreType.fa_dun_rot,
    "fa_dun_dev": pyrosetta.rosetta.core.scoring.ScoreType.fa_dun_dev,
    "fa_dun_semi": pyrosetta.rosetta.core.scoring.ScoreType.fa_dun_semi,
    "hxl_tors": pyrosetta.rosetta.core.scoring.ScoreType.hxl_tors,
    "ref": pyrosetta.rosetta.core.scoring.ScoreType.ref,
}


score_types = {
    "fa_atr": pyrosetta.rosetta.core.scoring.ScoreType.fa_atr,
    "fa_rep": pyrosetta.rosetta.core.scoring.ScoreType.fa_rep,
    "fa_sol": pyrosetta.rosetta.core.scoring.ScoreType.fa_sol,
    #    # "fa_intra_atr_xover4": pyrosetta.rosetta.core.scoring.ScoreType.fa_intra_atr_xover4,
    #    # "fa_intra_rep_xover4": pyrosetta.rosetta.core.scoring.ScoreType.fa_intra_rep_xover4,
    #    # "fa_intra_sol_xover4": pyrosetta.rosetta.core.scoring.ScoreType.fa_intra_sol_xover4,
    "lk_ball": pyrosetta.rosetta.core.scoring.ScoreType.lk_ball,
    #    # "lk_ball_iso": pyrosetta.rosetta.core.scoring.ScoreType.lk_ball_iso,
    #    # "lk_ball_bridge": pyrosetta.rosetta.core.scoring.ScoreType.lk_ball_bridge,
    #    # "lk_ball_bridge_uncpl": pyrosetta.rosetta.core.scoring.ScoreType.lk_ball_bridge_uncpl,
    "fa_elec": pyrosetta.rosetta.core.scoring.ScoreType.fa_elec,
    #    # "fa_intra_elec": pyrosetta.rosetta.core.scoring.ScoreType.fa_intra_elec,
    #    # "pro_close": pyrosetta.rosetta.core.scoring.ScoreType.pro_close,
    "hbond_sr_bb": pyrosetta.rosetta.core.scoring.ScoreType.hbond_sr_bb,
    "hbond_lr_bb": pyrosetta.rosetta.core.scoring.ScoreType.hbond_lr_bb,
    # "hbond_bb_sc": pyrosetta.rosetta.core.scoring.ScoreType.hbond_bb_sc,
    # "hb_sc": pyrosetta.rosetta.core.scoring.ScoreType.hbond_sc,
    #    # "dslf_fa13": pyrosetta.rosetta.core.scoring.ScoreType.dslf_fa13,
    #    # "rama_prepro": pyrosetta.rosetta.core.scoring.ScoreType.rama_prepro,
    #    # "omega": pyrosetta.rosetta.core.scoring.ScoreType.omega,
    #    # "p_aa_pp": pyrosetta.rosetta.core.scoring.ScoreType.p_aa_pp,
    #    # "fa_dun_rot": pyrosetta.rosetta.core.scoring.ScoreType.fa_dun_rot,
    #    # "fa_dun_dev": pyrosetta.rosetta.core.scoring.ScoreType.fa_dun_dev,
    #    # "fa_dun_semi": pyrosetta.rosetta.core.scoring.ScoreType.fa_dun_semi,
    #    # "hxl_tors": pyrosetta.rosetta.core.scoring.ScoreType.hxl_tors,
    #    # "ref": pyrosetta.rosetta.core.scoring.ScoreType.ref,
}


def pdbdata(pose, fname):

    sf = pyrosetta.rosetta.core.scoring.get_score_function()
    sfopt = sf.energy_method_options()
    sfopt.hbond_options().decompose_bb_hb_into_pair_energies(True)
    sf.set_energy_method_options(sfopt)
    sf(pose)

    energies = {
        "t_" + k: pose.energies().total_energies()[st]
        for k, st in all_score_types.items()
    }
    energiesw = {
        "t_w_" + k: pose.energies().weights()[st] * pose.energies().total_energies()[st]
        for k, st in all_score_types.items()
    }
    eweights = {k: pose.energies().weights()[st] for k, st in all_score_types.items()}
    # coords, etc

    ncac = get_bb_coords(pose)
    stubs = ncac_to_stubs(ncac).astype("f4")
    ncac = ncac
    cb = get_coords(pose, "CB")
    o = get_coords(pose, "O")
    com = np.mean(cb, axis=0)
    rg = np.sqrt(np.sum((cb - com) ** 2) / len(cb))
    coords = dict(ncac=ncac, cb=cb, o=o, stubs=stubs, com=com, rg=rg)
    chains = get_chain_bounds(pose)

    # one-body stuff

    resdata = dict(
        phi=[pose.phi(i) for i in range(1, len(pose) + 1)],
        psi=[pose.psi(i) for i in range(1, len(pose) + 1)],
        omega=[pose.omega(i) for i in range(1, len(pose) + 1)],
        chi1=[get_pose_chi(pose, i, 1) for i in range(1, len(pose) + 1)],
        chi2=[get_pose_chi(pose, i, 2) for i in range(1, len(pose) + 1)],
        chi3=[get_pose_chi(pose, i, 3) for i in range(1, len(pose) + 1)],
        chi4=[get_pose_chi(pose, i, 4) for i in range(1, len(pose) + 1)],
    )
    resdata = {k: np.array(v, "f4") for k, v in resdata.items()}
    for v in resdata.values():
        assert len(v) == len(pose)
    resdata["seq"] = pose.sequence()
    resdata["ss"] = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose).get_dssp_secstruct()
    resdata["chain"] = [pose.chain(i) - 1 for i in range(1, len(pose) + 1)]
    for k, st in all_score_types.items():
        tmp = list()
        for ir in range(len(pose)):
            tmp.append(pose.energies().residue_total_energies(ir + 1)[st])
        resdata["r_" + k] = np.array(tmp, np.float32)
    tmp = [pose.energies().residue_total_energy(i + 1) for i in range(len(pose))]
    resdata["r_etot"] = np.array(tmp, np.float32)

    # print(fname, "compute sasa")
    sasa_probe_vals = np.array([2, 3, 4])
    sasa = polya_sasa(pose, sasa_probe_vals)
    assert len(pose) == sasa.shape[0]
    assert sasa.shape[1] == len(sasa_probe_vals)
    for i, v in enumerate(sasa_probe_vals):
        resdata["sasa" + str(v)] = sasa[:, i]

    # two-body stuff
    chainseqs = [resdata["seq"][lb:ub] for lb, ub in chains]
    sym_chain_follows = [chainseqs.index(x) for x in chainseqs]

    pairdata = extract_pair_terms(**vars())

    return dict(
        fname=fname,
        coords=coords,
        chains=chains,
        resdata=resdata,
        pairdata=pairdata,
        energies=energies,
        energiesw=energiesw,
        eweights=eweights,
    )


def get_pose_chi(pose, ir, ichi):
    if ichi <= pose.residue(ir).nchi():
        return pose.chi(ichi, ir)
    return -12345.0


def extract_hbond_terms(pose, fname, **kw):
    hbset = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(pose, False, hbset)
    result = [dict(), dict(), dict(), dict()]
    for ihb in range(hbset.nhbonds()):
        hbond = hbset.hbond(ihb + 1)
        ir = hbond.don_res()
        jr = hbond.acc_res()
        irbb = hbond.don_hatm_is_protein_backbone()
        jrbb = hbond.acc_atm_is_protein_backbone()
        if jr < ir:
            ir, irbb, jr, bb = jr, jrbb, ir, irbb
        assert ir < jr
        idx = 2 * irbb + jrbb
        if (ir, jr) not in result[idx]:
            result[idx][ir, jr] = 0.0
        result[idx][ir, jr] += hbond.energy()

        # cute indexing sanity check...
        # print(irbb, jrbb, ["sc_sc", "sc_bb", "bb_sc", "bb_bb"][idx])
    # result = [np.array(list(x)) for x in result]
    labels = ["hb_sc_sc", "hb_sc_bb", "hb_bb_sc", "hb_bb_bb"]
    return {k: v for k, v in zip(labels, result)}


def extract_pair_terms(pose, sym_chain_follows, chains, fname, **kw):
    eweights = pose.energies().weights()
    energy_graph = pose.energies().energy_graph()
    hbonds = extract_hbond_terms(pose, fname)

    # print(fname, "extract pair energies")
    lbls = ["p_dist", "p_etot", "p_resi", "p_resj"]
    hb_lbls = ["hb_bb_bb", "hb_bb_sc", "hb_sc_bb", "hb_sc_sc"]
    lbls += ["p_" + l for l in hb_lbls]
    lbls += ["p_" + l for l in score_types]
    pairterms = {k: list() for k in lbls}
    for ichain, chain in enumerate(chains):
        if ichain != sym_chain_follows[ichain]:
            # print(fname, "skip sym redundant chain", ichain, "of", len(chains))
            continue
        for ir in range(*chain):
            assert pose.residue(ir + 1).is_protein()
            for jr in range(ir + 1, len(pose)):
                assert ir < jr
                edge = energy_graph.find_edge(ir + 1, jr + 1)
                if not edge:
                    continue
                etot = edge.dot(eweights)
                if etot == 0.0:
                    continue
                pairterms["p_resi"].append(ir)
                pairterms["p_resj"].append(jr)
                pairterms["p_dist"].append(np.sqrt(edge.square_distance()))
                pairterms["p_etot"].append(etot)
                for lbl in hb_lbls:
                    h = hbonds[lbl][(ir, jr)] if (ir, jr) in hbonds[lbl] else 0
                    pairterms["p_" + lbl].append(h)
                for lbl, st in score_types.items():
                    pairterms["p_" + lbl].append(edge[st])
    for k in pairterms:
        pairterms[k] = np.array(pairterms[k], "f4")
    for v in pairterms.values():
        assert len(v) == len(pairterms["p_dist"])
    return pairterms


def polya_sasa(pose, sasa_probe_vals):
    try:
        M = pyrosetta.rosetta.protocols.simple_moves.MakePolyXMover
    except:
        M = pyrosetta.rosetta.protocols.pose_creation.MakePolyXMover
    m = M("ALA", keep_pro=False, keep_gly=True, keep_disulfide_cys=True)
    polya_pose = pose.clone()
    m.apply(polya_pose)
    sasacalc = pyrosetta.rosetta.core.scoring.sasa.SasaCalc()

    rsdsasa = np.zeros((len(pose), len(sasa_probe_vals)), dtype="f4")
    for i, r in enumerate(sasa_probe_vals):
        sasacalc.set_probe_radius(r)
        sasacalc.calculate(polya_pose)
        rsdsasa[:, i] = np.array(sasacalc.get_residue_sasa())
        # print(
        # i,
        # r,
        # len(rsdsasa),
        # np.min(rsdsasa),
        # np.max(rsdsasa),
        # np.sum(rsdsasa[:, i] == 0),
        # )

    return rsdsasa


def get_bb_stubs(pose, which_resi=None):
    if which_resi is None:
        which_resi = list(range(1, pose.size() + 1))
    npstubs, n_ca_c = [], []
    for ir in which_resi:
        r = pose.residue(ir)
        if not r.is_protein():
            raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
        n, ca, c = r.xyz("N"), r.xyz("CA"), r.xyz("C")
        ros_stub = ros.core.kinematics.Stub(ca, n, ca, c)
        npstubs.append(numpy_stub_from_rosetta_stub(ros_stub))
        n_ca_c.append(np.array([[n.x, n.y, n.z], [ca.x, ca.y, ca.z], [c.x, c.y, c.z]]))
    return np.stack(npstubs).astype("f8"), np.stack(n_ca_c).astype("f8")


def get_bb_coords(pose, which_resi=None):
    if which_resi is None:
        which_resi = list(range(1, pose.size() + 1))
    n_ca_c = []
    for ir in which_resi:
        r = pose.residue(ir)
        if not r.is_protein():
            raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
        n, ca, c = r.xyz("N"), r.xyz("CA"), r.xyz("C")
        n_ca_c.append(
            np.array([[n.x, n.y, n.z, 1], [ca.x, ca.y, ca.z, 1], [c.x, c.y, c.z, 1]])
        )
    return np.stack(n_ca_c).astype("f4")


def get_coords(pose, aname, which_resi=None):
    if which_resi is None:
        which_resi = list(range(1, pose.size() + 1))
    xyzs = []
    for ir in which_resi:
        r = pose.residue(ir)
        if not r.is_protein():
            raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
        if r.has(aname):
            xyz = r.xyz(aname)
        else:
            xyz = r.xyz("CA")
        xyzs.append(np.array([xyz.x, xyz.y, xyz.z, 1]))
    return np.stack(xyzs).astype("f4")


def get_chain_bounds(pose):
    ch = np.array([pose.chain(i + 1) for i in range(len(pose))])
    chains = list()
    for i in range(ch[-1]):
        chains.append((np.sum(ch <= i), np.sum(ch <= i + 1)))
    assert chains[0][0] == 0
    assert chains[-1][-1] == len(pose)
    return chains


def ncac_to_stubs(ncac):
    """
        Vector const & center,
        Vector const & a,
        Vector const & b,
        Vector const & c
    )
    {
        Vector e1( a - b);
        e1.normalize();

        Vector e3( cross( e1, c - b ) );
        e3.normalize();

        Vector e2( cross( e3,e1) );
        M.col_x( e1 ).col_y( e2 ).col_z( e3 );
        v = center;
    """
    assert ncac.shape[1:] == (3, 4)
    stubs = np.zeros((len(ncac), 4, 4), dtype=np.float64)
    ca2n = (ncac[:, 0] - ncac[:, 1])[..., :3]
    ca2c = (ncac[:, 2] - ncac[:, 1])[..., :3]
    # tgt1 = ca2n + ca2c  # thought this might make
    # tgt2 = ca2n - ca2c  # n/c coords match better
    tgt1 = ca2n  # rosetta style
    tgt2 = ca2c  # seems better
    a = tgt1
    a /= np.linalg.norm(a, axis=-1)[:, None]
    c = np.cross(a, tgt2)
    c /= np.linalg.norm(c, axis=-1)[:, None]
    b = np.cross(c, a)
    assert np.allclose(np.sum(a * b, axis=-1), 0, atol=1e-6)
    assert np.allclose(np.sum(b * c, axis=-1), 0, atol=1e-6)
    assert np.allclose(np.sum(c * a, axis=-1), 0, atol=1e-6)
    assert np.allclose(np.linalg.norm(a, axis=-1), 1, atol=1e-6)
    assert np.allclose(np.linalg.norm(b, axis=-1), 1, atol=1e-6)
    assert np.allclose(np.linalg.norm(c, axis=-1), 1, atol=1e-6)
    stubs[:, :3, 0] = a
    stubs[:, :3, 1] = b
    stubs[:, :3, 2] = c
    stubs[:, :3, 3] = ncac[:, 1, :3]
    stubs[:, 3, 3] = 1
    return stubs
