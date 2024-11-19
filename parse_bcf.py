from pybcf import BcfReader
from collections import defaultdict
import numpy as np
import moments
import pickle


def get_population_samples(samples_path):
    samples = defaultdict(list)
    with open(samples_path, "r") as fin:
        for line in fin:
            if line.startswith("CAAPA_ID"):
                continue
            else:
                ID, POP, _, _ = line.split("\t")
                samples[POP].append(ID)
    return samples


def count_derived_alleles(genotypes, sample_idx, pop):
    alleles, num = np.unique(genotypes[sample_idx[pop]], return_counts=True)
    num = num[np.isnan(alleles) == False]
    alleles = alleles[np.isnan(alleles) == False]
    if len(alleles) > 1:
        n = int(np.sum(num))
        i = int(np.sum(num[alleles != 0]))
        return (n, i)
    else:
        return None


def get_single_populations_sfs(bcf_path, samples):
    bcf = BcfReader("CAAPA_Freeze2_NAM_MSL_CBU_TWA_BAK_chr22.bcf")
    # get population sample indexes
    sample_idx = {
        pop: [bcf.samples.index(s) for s in ids] for pop, ids in samples.items()
    }
    # tally allele counts, storing (n, i) as keys
    # here, n is the sample size (accounting for missing data), and i is the
    # number of derived alleles
    counts = {pop: defaultdict(int) for pop in samples.keys()}
    for var in bcf:
        genotypes = var.samples["GT"]
        for pop in samples.keys():
            k = count_derived_alleles(genotypes, sample_idx, pop)
            if k is not None:
                counts[pop][k] += 1
    return counts


def build_spectra(counts):
    spectra = {p: {} for p in samples.keys()}
    for p in samples.keys():
        for k, v in counts[p].items():
            n, i = k
            if n not in spectra[p]:
                spectra[p][n] = np.zeros(n + 1)
            spectra[p][n][i] = v
    return spectra


def project_spectra(spectra, pop, nmin):
    fs = moments.Spectrum(np.zeros(nmin + 1))
    for n, fs_from in spectra[pop].items():
        if n >= nmin:
            fs += moments.Spectrum(fs_from).project([nmin])
    return fs

if __name__ == "__main__":
    sample_path = "samples_metadata.txt"
    samples = get_population_samples(sample_path)

    # count alleles in the VCF
    bcf_path = "CAAPA_Freeze2_NAM_MSL_CBU_TWA_BAK_chr22.bcf"
    counts = get_single_populations_sfs(bcf_path, samples)

    # compile the spectra from the parsed counts
    spectra = build_spectra(counts)
    with open("all_1d_spectra.pkl", "wb+") as fout:
        pickle.dump(spectra, fout)

    # sample sizes are selected so that pi(n) / pi(max) is > 99.5%
    nmax = {"Baka": 56, "Mende": 156, "Nama": 70, "BaTwa": 94, "Chabu": 42}
    spectra_proj = {
        pop: project_spectra(spectra, pop, nmax[pop]) for pop in nmax.keys()
    }
    for pop in spectra_proj.keys():
        spectra_proj[pop].pop_ids = [pop]

    with open("projected_1d_spectra.pkl", "wb+") as fout:
        pickle.dump(spectra_proj, fout)
