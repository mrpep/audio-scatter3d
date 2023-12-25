from encodecmae import load_model as load_encodecmae
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from encodecgpt.tasks.models import EnCodecGPT
from tqdm import tqdm

gpt_models_metadata = {'Librispeech': {'hf_path': 'lpepino/encodecgpt/librilight-base/librilight-500ksteps-base.ckpt',
                                   'encodecmae_model': 'base',
                                   'context': 3},
                    'NSynth': {'hf_path': 'lpepino/encodecgpt/nsynth-base/nsynth-390ksteps-base.ckpt',
                              'encodecmae_model': 'base',
                              'context': 3},
                    'Audioset': {'hf_path': 'lpepino/encodecgpt/audioset-base-st/audioset-500ksteps-base-st.ckpt',
                                'encodecmae_model': 'base-st',
                                'context': 10}}

encodecmae_models_metadata = {'base': {'hf_path': 'lpepino/encodecmae-base/model.pt'},
                              'base-st': {'hf_path': 'lpepino/encodecmae-base-st/model.pt'}}


def roll_along(arr, shifts, dim):
    assert arr.ndim - 1 == shifts.ndim
    dim %= arr.ndim
    shape = (1,) * dim + (-1,) + (1,) * (arr.ndim - dim - 1)
    dim_indices = torch.arange(arr.shape[dim], device=arr.device).reshape(shape)
    indices = (dim_indices - shifts.unsqueeze(dim)) % arr.shape[dim]
    
    return torch.gather(arr, dim, indices)

def apply_delay_pattern(codes):
    codes = torch.nn.functional.pad(codes+1,(0,codes.shape[1]-1))
    codes = roll_along(codes, torch.arange(0,codes.shape[1], device=codes.device)[None,:].tile((codes.shape[0],1)), 2)
    return codes

def generate_interp(prompts, model, buffer_size=300, temperature=0.7, generation_steps=500, initial_values=None, cache=True):
    if prompts.ndim==1:
        prompts = np.tile(prompts[np.newaxis,:],(generation_steps,1))
        cache = True
    prompts = torch.from_numpy(prompts).to(model.device, dtype=model.dtype)
    if initial_values is not None:
        with torch.no_grad():
            initial_values = model.encodec_model.encode(torch.from_numpy(initial_values).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32,device=model.device))[0][0][:,:model.num_q,:]
            initial_values = apply_delay_pattern(initial_values)
            initial_values = initial_values[:,:,:-model.num_q+1]
            input_vocab_embed = torch.arange(model.num_q, device=model.device)[None,:,None]*(model.num_codes + 1) + initial_values
            gpt_in = model.vocab_embed(input_vocab_embed).sum(axis=1)
            gpt_in = torch.cat([prompts[0,None,:].unsqueeze(0),gpt_in],axis=1)
            generation = [p[:,0] for p in initial_values.transpose(0,2)]
            start_i = gpt_in.shape[1] - 1
    else:
        gpt_in = prompts[0].unsqueeze(0)
        start_i = 0
        generation = []
    if not isinstance(temperature, np.ndarray):
        temperature = temperature*np.ones((len(prompts) + len(generation),))
    past_kv = None
    with torch.no_grad():
        for j,p in tqdm(enumerate(prompts)):
            i = start_i + j
            outs = model.gpt(inputs_embeds=gpt_in, past_key_values=past_kv, use_cache=True)
            past_kv = outs['past_key_values']
            
            preds = model.classification_head(outs['last_hidden_state'])
            preds = preds.view(preds.shape[0],model.num_q,model.num_codes+1)
            
            sampled_idxs = torch.cat([torch.multinomial(torch.nn.functional.softmax(preds[0,q,:]/temperature[i]),1) for q in range(model.num_q)])
            generation.append(sampled_idxs)
            #if i<buffer_size:
            in_idxs = torch.arange(model.num_q, device=model.device)*(model.num_codes + 1) + sampled_idxs
            gpt_in = model.vocab_embed(in_idxs).sum(axis=0).unsqueeze(0)
            #else:
            #    generation_ = torch.stack(generation)[-buffer_size:]
            #    for k in range(model.num_q-1):
            #        generation_[k,k+1:] = 0
            #    in_idxs = (torch.arange(model.num_q, device=model.device)[None,:])*(model.num_codes + 1) + generation_
            #    gpt_seq = model.vocab_embed(in_idxs).sum(axis=1).unsqueeze(0)
            #    gpt_in = torch.cat([gpt_in[:,0].unsqueeze(1), gpt_seq],axis=1)
                    
        generation = torch.stack(generation)
        print(generation.shape)
        generation = roll_along(generation,-torch.arange(0,8,device=generation.device),0)
        
        audio = model.encodec_model.decode([(torch.maximum(generation-1, torch.tensor(0, device=model.device))[:-model.num_q].T.unsqueeze(0),None)])
        audio = audio[0].cpu().detach().numpy()
        return audio, generation
    
def load_models():
    state = {}
    state['device'] = 'cuda:0'
    print('Loading models...')
    device = state['device']
    gpt_state_dicts = {}
    encodecmae_state_dicts = {}
    encodecmae_base = load_encodecmae('base', device=device)
    encodecmae_base.visible_encoder.compile=False
    egpt = EnCodecGPT()
    egpt.to(device)
    for k,v in gpt_models_metadata.items():
        hf_repo = '/'.join(v['hf_path'].split('/')[:2])
        hf_filename = '/'.join(v['hf_path'].split('/')[2:])
        ckpt_file = hf_hub_download(repo_id=hf_repo,filename=hf_filename)
        gpt_state_dicts[k] = torch.load(ckpt_file, map_location=device)['state_dict']
    for k,v in encodecmae_models_metadata.items():
        hf_repo = '/'.join(v['hf_path'].split('/')[:2])
        hf_filename = '/'.join(v['hf_path'].split('/')[2:])
        ckpt_file = hf_hub_download(repo_id=hf_repo,filename=hf_filename)        
        encodecmae_state_dicts[k] = torch.load(ckpt_file, map_location=device)['state_dict']
    
    state['gpt_sd'], state['emae_sd'], state['gpt'], state['emae'] = gpt_state_dicts, encodecmae_state_dicts, egpt, encodecmae_base
    return state