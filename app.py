import os
from flask import Flask, render_template, request, url_for, redirect
import datahelper
import tensorflow as tf
import numpy as np
import logging
import threading
from urllib.parse import unquote_plus
from rdkit import Chem
from rdkit.Chem import Draw
import base64
from io import BytesIO

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Thread-safe model loading
model_lock = threading.Lock()
combined_model = None
single_model = None

def load_models():
    global combined_model, single_model
    with model_lock:
        try:
            combined_model = tf.keras.models.load_model("models/combined_kiba.h5", compile=False)
            single_model = tf.keras.models.load_model("models/single_prot_kiba.h5", compile=False)
            logger.debug("Models loaded. Combined input shape: %s, Single input shape: %s", 
                         combined_model.input_shape, single_model.input_shape)
            for layer in combined_model.layers:
                if hasattr(layer, 'get_weights'):
                    weights = layer.get_weights()
                    logger.debug("Layer %s has weights: %s", layer.name, "Yes" if weights else "No")
            return True
        except Exception as e:
            logger.error("Failed to load models: %s", str(e))
            return False

MAX_SMI_LEN = 100
MAX_SEQ_LEN = 1000
weight_combined = 0.4
weight_single = 0.6

def validate_smiles(smiles, charset):
    logger.debug("Validating SMILES: %s, type: %s, charset keys: %s", smiles, type(smiles), list(charset.keys()))
    valid = all(char in charset for char in smiles)
    logger.debug("Validation result: %s", valid)
    return valid

def validate_protein(protein, charset):
    return all(char in charset for char in protein) and len(protein) >= 50

def minmax_norm(val, min_val=0, max_val=17):
    return (val - min_val) / (max_val - min_val)

def inverse_minmax_norm(norm_val, min_val=0, max_val=17):
    return norm_val * (max_val - min_val) + min_val

def predict_affinity(smiles, protein, weighted_model, combined_model, single_model):
    if not validate_smiles(smiles, datahelper.CHARISOSMISET) or not validate_protein(protein, datahelper.CHARPROTSET):
        raise ValueError("Invalid SMILES or protein sequence.")
    
    X_smi = datahelper.label_smiles(smiles, MAX_SMI_LEN, datahelper.CHARISOSMISET)
    X_prot = datahelper.label_sequence(protein, MAX_SEQ_LEN, datahelper.CHARPROTSET)
    X_smi = np.reshape(X_smi, (1, MAX_SMI_LEN))
    X_prot = np.reshape(X_prot, (1, MAX_SEQ_LEN))

    # Use input names if available
    input_names = [inp.name.split(':')[0] for inp in combined_model.input]
    logger.debug("Model input names: %s", input_names)
    input_dict = {input_names[0]: X_smi, input_names[1]: X_prot}
    
    y_combined = float(combined_model.predict(input_dict)[0][0])
    y_single = float(single_model.predict(input_dict)[0][0])

    norm_combined = minmax_norm(y_combined)
    norm_single = minmax_norm(y_single)
    ensemble_norm = weight_combined * norm_combined + weight_single * norm_single
    final_affinity = inverse_minmax_norm(ensemble_norm)

    return final_affinity, y_combined, y_single

@app.route('/')
def title():
    return render_template('title.html')

@app.route('/go')
def go():
    return redirect(url_for('index'))

@app.route('/index')
def index():
    error = None
    return render_template('index.html', error=error)

@app.route('/index', methods=['POST'])
def predict():
    global combined_model, single_model
    smiles = request.form.get('smiles')
    protein_seq = request.form.get('protein')
    error = None

    if not smiles or not protein_seq:
        error = "Please provide both SMILES and protein sequence."
    else:
        # Reload models per request to ensure fresh state
        if not load_models():
            error = "Failed to load prediction models. Please check model files or contact support."
            return render_template('index.html', error=error)

        try:
            with tf.device('/cpu:0'):  # Optional CPU forcing
                weighted_aff, combined_aff, single_aff = predict_affinity(smiles, protein_seq, None, combined_model, single_model)
                props = datahelper.calculate_ligand_properties(smiles) if smiles else [None] * 12
                mw, logp, h_donors, h_acceptors, rotatable_bonds, tpsa = props[1], props[4], props[6], props[7], props[8], props[5]
                logger.debug("Extracted properties: mw=%s, logp=%s, h_donors=%s, h_acceptors=%s, rotatable_bonds=%s, tpsa=%s",
                             mw, logp, h_donors, h_acceptors, rotatable_bonds, tpsa)
                return redirect(url_for('result', smiles=smiles, protein=protein_seq, weighted_aff=weighted_aff,
                                        combined_aff=combined_aff, single_aff=single_aff, mw=mw, logp=logp,
                                        h_donors=h_donors, h_acceptors=h_acceptors, rotatable_bonds=rotatable_bonds, tpsa=tpsa))
        except Exception as e:
            error = f"Prediction error: {str(e)}"
            # Attempt to reload models on failure
            if not load_models():
                error += " Failed to reload models."
    return render_template('index.html', error=error)

@app.route('/result')
def result():
    smiles = request.args.get('smiles')
    protein = request.args.get('protein')
    weighted_aff = request.args.get('weighted_aff')
    combined_aff = request.args.get('combined_aff')
    single_aff = request.args.get('single_aff')
    return render_template('result.html', smiles=smiles, protein=protein, weighted_aff=weighted_aff,
                           combined_aff=combined_aff, single_aff=single_aff)

@app.route('/ligand_check')
def ligand_check():
    smiles = request.args.get('smiles')
    if not smiles:
        return render_template('ligand_check.html', error="No SMILES provided.")
    try:
        decoded_smiles = unquote_plus(smiles)
        logger.debug("Decoded SMILES: %s", decoded_smiles)
        if not validate_smiles(decoded_smiles, datahelper.CHARISOSMISET):
            raise ValueError("SMILES contains invalid characters.")
        mol = Chem.MolFromSmiles(decoded_smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string after decoding.")
        props = datahelper.calculate_ligand_properties(decoded_smiles)
        logger.debug("Calculated properties: %s", props)
        lipinski = datahelper.check_lipinski_rule(props[1], props[4], props[6], props[7]) if all(p is not None for p in [props[1], props[4], props[6], props[7]]) else False
        logger.debug("Lipinski rule params: %s", [props[1], props[4], props[6], props[7]])
        veber = datahelper.check_veber_rule(props[8], props[5]) if props[8] is not None and props[5] is not None else False
        pains = datahelper.check_pains(decoded_smiles)
        brenk = datahelper.check_brenk(decoded_smiles)
        img = Draw.MolToImage(mol, size=(300, 300))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return render_template('ligand_check.html', smiles=decoded_smiles, props=props, lipinski=lipinski,
                               veber=veber, pains=pains, brenk=brenk, ligand_image=img_str)
    except Exception as e:
        logger.error("Ligand check error: %s", str(e), exc_info=True)
        return render_template('ligand_check.html', error=f"Error processing SMILES: {str(e)}", props=[None] * 12)

@app.route('/protein_analysis')
def protein_analysis():
    protein = request.args.get('protein')
    if not protein:
        return render_template('protein_display.html', error="No protein sequence provided.", protein=None)

    try:
        # Use the enhanced analyze_protein from datahelper (includes InterPro + DeepTMHMM)
        protein_data = datahelper.analyze_protein(protein)

        return render_template('protein_display.html', protein=protein_data, error=None)

    except Exception as e:
        logging.error(f"Protein analysis error: {e}", exc_info=True)
        return render_template('protein_display.html', error=f"Error analyzing protein: {str(e)}", protein=None)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
