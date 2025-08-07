import numpy as np

# Max sequence lengths
MAX_SMI_LEN = 100
MAX_SEQ_LEN = 1000

# Ensemble weights
weight_combined = 0.4
weight_single = 0.6

# Character sets for SMILES and protein sequences
CHARISOSMISET = {
    'C': 1, 'c': 2, 'N': 3, 'n': 4, 'O': 5, 'o': 6, '=': 7, '#': 8, '(': 9, ')': 10,
    '1': 11, '2': 12, '3': 13, '4': 14, '5': 15, '6': 16, '[': 17, ']': 18,
    '-': 19, '@': 20, '/': 21, '\\': 22, 'F': 23, 'Cl': 24, 'Br': 25, 'S': 26
}
CHARPROTSET = {c: i + 1 for i, c in enumerate('ACDEFGHIKLMNPQRSTVWY')}

def predict_affinity(smiles, protein,vi, combined_model, single_model):
    """
    Predict binding affinity using ensemble of combined_model and single_model.
    """
    if not validate_smiles(smiles, CHARISOSMISET):
        raise ValueError("Invalid SMILES: Contains invalid characters or syntax")
    if not validate_protein(protein, CHARPROTSET):
        raise ValueError("Invalid protein sequence: Contains invalid amino acids or too short")

    # Preprocess SMILES and protein sequence
    X_smi = label_smiles(smiles, MAX_SMI_LEN, CHARISOSMISET)
    X_prot = label_sequence(protein, MAX_SEQ_LEN, CHARPROTSET)
    X_smi = np.reshape(X_smi, (1, MAX_SMI_LEN))
    X_prot = np.reshape(X_prot, (1, MAX_SEQ_LEN))

    # Run predictions
    try:
        y_combined = float(combined_model.predict([X_smi, X_prot], verbose=0)[0][0])
        y_single = float(single_model.predict([X_smi, X_prot], verbose=0)[0][0])
    except Exception as e:
        raise ValueError(f"Model prediction failed: {str(e)}")

    # Normalize predictions for weighted ensemble
    norm_combined = minmax_norm(y_combined)
    norm_single = minmax_norm(y_single)
    ensemble_norm = weight_combined * norm_combined + weight_single * norm_single
    final_affinity = inverse_minmax_norm(ensemble_norm)

    return final_affinity, y_combined, y_single

def validate_smiles(smiles, charset):
    """Validate SMILES characters."""
    return all(char in charset for char in smiles)

def validate_protein(protein, charset):
    """Validate protein sequence (chars + length)."""
    return all(char in charset for char in protein) and len(protein) >= 50

def label_smiles(smiles, max_len, charset):
    """Convert SMILES string to numerical array."""
    X = np.zeros(max_len)
    for i, c in enumerate(smiles[:max_len]):
        X[i] = charset.get(c, 0)
    return X

def label_sequence(sequence, max_len, charset):
    """Convert protein sequence to numerical array."""
    X = np.zeros(max_len)
    for i, c in enumerate(sequence[:max_len]):
        X[i] = charset.get(c, 0)
    return X

def minmax_norm(val, min_val=0, max_val=17):
    """Normalize value to [0,1] range."""
    return (val - min_val) / (max_val - min_val)

def inverse_minmax_norm(norm_val, min_val=0, max_val=17):
    """Convert normalized value back to original range."""
    return norm_val * (max_val - min_val) + min_val