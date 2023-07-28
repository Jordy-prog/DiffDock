import copy
import os
import torch
from argparse import ArgumentParser, Namespace
from rdkit.Chem import RemoveHs
from functools import partial
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch_geometric.loader import DataLoader

from datasets.process_mols import write_mol_with_coords
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.inference_utils_JS import InferenceDataset, set_nones, preprocess_data, postprocess_data
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.visualise import PDBFile
from tqdm import tqdm
import warnings

RDLogger.DisableLog('rdApp.*')
import yaml
parser = ArgumentParser()
parser.add_argument('--protein_ligand_csv', type=str, default=None, required=True, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path, --protein_sequence and --ligand parameters')
parser.add_argument('--samples_per_complex', type=int, default=3, help='Number of samples to generate')

parser.add_argument('--model_dir', type=str, default='workdir/paper_score_model', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
parser.add_argument('--confidence_model_dir', type=str, default='workdir/paper_confidence_model', help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')

parser.add_argument('--batch_size', type=int, default=10, help='')
parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--inference_steps', type=int, default=10, help='Number of denoising steps')
parser.add_argument('--actual_steps', type=int, default=10, help='Number of denoising steps that are actually performed')
args = parser.parse_args()

# Function for splitting dataframe (temporary, I hope)
def split_dataframe(df, chunk_size=1000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

args.out_dir = 'results'
args.out_file_base = os.path.join(args.out_dir, args.protein_ligand_csv.split('/')[-1].replace('.csv', ''))

# Load model parameters
with open(f'{args.model_dir}/model_parameters.yml') as f:
    score_model_args = Namespace(**yaml.full_load(f))
if args.confidence_model_dir is not None:
    with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load csv and split the data into chunks per klifs ID
klifs_chunks = preprocess_data(args.protein_ligand_csv)

# Execute inference for each klifs ID
for klifs, klifs_chunk in klifs_chunks:
    print('Docking in KLIFS:', klifs)

    final_klifs_data = pd.DataFrame({'klifs_ID': [], 'SMILES_input': [], 'SMILES_output': [], 'molfile_compressed': [], 'DiffDock_confidence': []})

    chunks = split_dataframe(klifs_chunk) # Split data in chunks of 1000 complexes to avoid memory issues

    for chunk in chunks:
        # remove nan strings
        complex_name_list = set_nones(chunk['complex_name'].tolist())
        protein_path_list = set_nones(chunk['protein_path'].tolist())
        protein_sequence_list = set_nones(chunk['protein_sequence'].tolist())
        ligand_description_list = set_nones(chunk['ligand_description'].tolist())

        complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]

        # preprocessing of complexes into geometric graphs
        test_dataset = InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                                        ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                                        lm_embedding=score_model_args.esm_embeddings_path is not None,
                                        receptor_radius=score_model_args.receptor_radius, remove_hs=score_model_args.remove_hs,
                                        c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                                        all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
                                        atom_max_neighbors=score_model_args.atom_max_neighbors)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        if args.confidence_model_dir is not None and not confidence_args.use_original_model_cache:
            print('HAPPENING | confidence model uses different type of graphs than the score model. '
                'Loading (or creating if not existing) the data for the confidence model now.')
            confidence_test_dataset = \
                InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                                ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                                lm_embedding=confidence_args.esm_embeddings_path is not None,
                                receptor_radius=confidence_args.receptor_radius, remove_hs=confidence_args.remove_hs,
                                c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                                all_atoms=confidence_args.all_atoms, atom_radius=confidence_args.atom_radius,
                                atom_max_neighbors=confidence_args.atom_max_neighbors,
                                precomputed_lm_embedding=test_dataset.lm_embedding)
        else:
            confidence_test_dataset = None

        t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

        with warnings.catch_warnings(): # Ignore warnings from the model
            warnings.simplefilter("ignore")
            model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)

        state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()

        if args.confidence_model_dir is not None:
            with warnings.catch_warnings(): # Ignore warnings from the model
                warnings.simplefilter("ignore")
                confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True)

            state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
            confidence_model.load_state_dict(state_dict, strict=True)
            confidence_model = confidence_model.to(device)
            confidence_model.eval()
        else:
            confidence_model = None
            confidence_args = None

        tr_schedule = get_t_schedule(inference_steps=args.inference_steps)

        failures, skipped = 0, 0
        N = args.samples_per_complex
        print('Size of test dataset: ', len(test_dataset))
        for idx, orig_complex_graph in tqdm(enumerate(test_loader)): # SMILES == complex_graph['name']
            if not orig_complex_graph.success[0]:
                skipped += 1
                print(f"HAPPENING | The test dataset did not contain {test_dataset.complex_names[idx]} for {test_dataset.ligand_descriptions[idx]} and {test_dataset.protein_files[idx]}. We are skipping this complex.")
                continue
            try:
                if confidence_test_dataset is not None:
                    confidence_complex_graph = confidence_test_dataset[idx]
                    if not confidence_complex_graph.success:
                        skipped += 1
                        print(f"HAPPENING | The confidence dataset did not contain {orig_complex_graph.name}. We are skipping this complex.")
                        continue
                    confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
                else:
                    confidence_data_list = None
                data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
                randomize_position(data_list, score_model_args.no_torsion, False, score_model_args.tr_sigma_max)
                lig = orig_complex_graph.mol[0]

                # run reverse diffusion
                data_list, confidence = sampling(data_list=data_list, model=model,
                                                inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                                tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule,
                                                device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                                confidence_model=confidence_model,
                                                confidence_data_list=confidence_data_list, confidence_model_args=confidence_args,
                                                batch_size=args.batch_size, no_final_step_noise=args.no_final_step_noise)
                ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])

                # reorder predictions based on confidence output
                if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
                    confidence = confidence[:, 0]
                if confidence is not None:
                    confidence = confidence.cpu().numpy()
                    re_order = np.argsort(confidence)[::-1]
                    confidence = confidence[re_order]
                    ligand_pos = ligand_pos[re_order]

                # postprocess results and append to final dataframe
                klifs_data_chunk = postprocess_data(copy.deepcopy(lig), ligand_pos, klifs, orig_complex_graph['name'][0], confidence, remove_hs=score_model_args.remove_hs)
                final_klifs_data = pd.concat([final_klifs_data, klifs_data_chunk])  

            except Exception as e:
                print("Failed on", orig_complex_graph["name"], e)
                failures += 1

    print(f'Failed for {failures} complexes')
    print(f'Skipped {skipped} complexes')
    print(f'Results are in {args.out_dir}')

    final_klifs_data.to_csv(f'{args.out_file_base}_{klifs}_results.csv', index=False)

