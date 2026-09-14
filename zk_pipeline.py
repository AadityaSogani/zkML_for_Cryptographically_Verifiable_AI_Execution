import os
import ezkl

def run_zk_pipeline():
    os.makedirs("circuits", exist_ok=True)
    os.makedirs("proofs", exist_ok=True)

    model_path = "models/model.onnx"
    data_path = "data/input.json"
    settings_path = "circuits/settings.json"
    compiled_circuit_path = "circuits/network.compiled"
    pk_path = "circuits/pk.key"
    vk_path = "circuits/vk.key"
    witness_path = "proofs/witness.json"
    proof_path = "proofs/proof.json"

    print("1. Generating EZKL settings configuration...")
    py_args = ezkl.PyRunArgs()
    py_args.input_visibility = "public"
    py_args.param_visibility = "fixed"
    py_args.output_visibility = "public"
    
    ezkl.gen_settings(model_path, settings_path, py_run_args=py_args)

    print("2. Calibrating circuit settings...")
    ezkl.calibrate_settings(data_path, model_path, settings_path, "resources")

    print("3. Compiling the ZK circuit...")
    ezkl.compile_circuit(model_path, compiled_circuit_path, settings_path)

    print("4. Fetching Structured Reference String (SRS)...")
    ezkl.get_srs(settings_path)

    print("5. Setting up proving and verification keys...")
    ezkl.setup(compiled_circuit_path, vk_path, pk_path)

    print("6. Generating Witness (executing the model inside the circuit)...")
    ezkl.gen_witness(data_path, compiled_circuit_path, witness_path)

    print("7. Generating Cryptographic Zero-Knowledge Proof...")
    ezkl.prove(witness_path, compiled_circuit_path, pk_path, proof_path, "single")

    print("8. Verifying the Zero-Knowledge Proof...")
    res = ezkl.verify(proof_path, settings_path, vk_path)
    
    print(f"\nVerification Result: {res}")
    if res:
        print("SUCCESS! The AI execution was cryptographically verified without exposing model weights.")

if __name__ == "__main__":
    run_zk_pipeline()