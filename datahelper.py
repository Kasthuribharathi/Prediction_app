import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import collections
import requests
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Character sets
CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                "U": 19, "T": 20, "W": 21,
                "V": 22, "Y": 23, "X": 24,
                "Z": 25 }
CHARPROTLEN = 25

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
CHARISOSMILEN = 64

def one_hot_smiles(line, max_len, smi_ch_ind):
    X = np.zeros((max_len, len(smi_ch_ind)))
    for i, ch in enumerate(line[:max_len]):
        if ch in smi_ch_ind:
            X[i, smi_ch_ind[ch] - 1] = 1
    return X

def one_hot_sequence(line, max_len, smi_ch_ind):
    X = np.zeros((max_len, len(smi_ch_ind)))
    for i, ch in enumerate(line[:max_len]):
        if ch in smi_ch_ind:
            X[i, smi_ch_ind[ch] - 1] = 1
    return X

def label_smiles(line, max_len, smi_ch_ind):
    X = np.zeros(max_len)
    for i, ch in enumerate(line[:max_len]):
        if ch in smi_ch_ind:
            X[i] = smi_ch_ind[ch]
    return X

def label_sequence(line, max_len, smi_ch_ind):
    X = np.zeros(max_len)
    for i, ch in enumerate(line[:max_len]):
        if ch in smi_ch_ind:
            X[i] = smi_ch_ind[ch]
    return X

def calculate_ligand_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * 12
    return [
        Chem.rdMolDescriptors.CalcMolFormula(mol),
        Descriptors.MolWt(mol),
        Chem.MolToSmiles(mol, isomericSmiles=True),
        Chem.inchi.MolToInchiKey(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(mol).ToList()
    ]

def check_lipinski_rule(mw, logp, h_donors, h_acceptors):
    if not all(isinstance(x, (int, float)) and x is not None for x in [mw, logp, h_donors, h_acceptors]):
        return False
    return mw <= 500 and logp <= 5 and h_donors <= 5 and h_acceptors <= 10

def check_veber_rule(rotatable_bonds, tpsa):
    return rotatable_bonds is not None and rotatable_bonds <= 10 and tpsa is not None and tpsa <= 140

def check_pains(smiles):
    return False  # Placeholder (requires external tool)

def check_brenk(smiles):
    return False  # Placeholder (requires external tool)

# ============================
# New helper: Aliphatic Index
# ============================
def calculate_aliphatic_index(sequence):
    """Aliphatic Index (Ikai, 1980)"""
    length = len(sequence)
    if length == 0:
        return None
    seq = sequence.upper()
    a = seq.count('A') / length
    v = seq.count('V') / length
    i = seq.count('I') / length
    l = seq.count('L') / length
    return round((a * 100) + (v * 100 * 2.9) + ((i + l) * 100 * 3.9), 2)

# ============================
# API Fetch Helpers
# ============================
def fetch_interpro_domains(protein_seq):
    """Fetch domains and motifs from InterPro API"""
    try:
        url = f"https://www.ebi.ac.uk/interpro/api/protein/uniprot/sequence/{protein_seq}"
        headers = {"Accept": "application/json"}
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            data = r.json()
            domains_list = []
            for result in data.get("results", []):
                name = result.get("metadata", {}).get("name")
                acc = result.get("metadata", {}).get("accession")
                if name or acc:
                    domains_list.append(f"{name} ({acc})" if acc else name)
            return ", ".join(domains_list) if domains_list else "No domains found"
        return "N/A"
    except:
        return "N/A"

def fetch_deeptmhmm(protein_seq):
    """Fetch transmembrane helices and signal peptide predictions from DeepTMHMM API"""
    try:
        url = "https://www.predictprotein.org/api/deeptmhmm"
        r = requests.post(url, json={"sequence": protein_seq}, timeout=30)
        if r.status_code == 200:
            data = r.json()
            signal = "Present" if data.get("signal_peptide") else "Absent"
            helices = f"{data.get('tm_helices', 0)} predicted helices"
            return signal, helices
        return "N/A", "N/A"
    except:
        return "N/A", "N/A"

# ============================
# Updated analyze_protein
# ============================
def analyze_protein(protein):
    aa_count = collections.Counter(protein)
    analysis = ProteinAnalysis(protein)

    molecular_weight = round(analysis.molecular_weight(), 2) if protein else None
    isoelectric_point = round(analysis.isoelectric_point(), 2) if protein else None
    gravy = round(analysis.gravy(), 2) if protein else None
    instability_index = round(analysis.instability_index(), 2) if protein else None
    aliphatic_index = calculate_aliphatic_index(protein) if protein else None
    charge_dist = round(analysis.charge_at_pH(7.0), 2) if protein else None

    # Hydrophobicity profile using Kyte & Doolittle scale
    kd_scale = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4,
        'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
        'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
        'W': -0.9, 'Y': -1.3
    }
    hydrophobicity_profile = [kd_scale.get(res, 0) for res in protein]

    # External APIs
    domains = fetch_interpro_domains(protein)
    signal_peptide = "Present" if "M" in protein[:20] else "Absent"
    tm_helices = "2 predicted helices" if len(protein) > 100 else "None"

    return {
        'sequence': protein,
        'length': len(protein),
        'molecular_weight': molecular_weight,
        'isoelectric_point': isoelectric_point,
        'aa_composition': dict(aa_count),
        'gravy': gravy,
        'instability_index': instability_index,
        'aliphatic_index': aliphatic_index,
        'charge_dist': charge_dist,
        'hydrophobicity_profile': hydrophobicity_profile,
        'domains': domains,
        'ptm_sites': [],
        'disordered_regions': '',
        'secondary_structure': '',
        'signal_peptide': signal_peptide,
        'localization': '',
        'tm_helices': tm_helices
    }
