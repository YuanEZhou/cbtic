# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
import pdb
from .CaptionModel import CaptionModel
from .AttModel import pack_wrapper, AttModel
from .CB_AttModel import  CB_AttModel

class CB_AttEnsemble(CB_AttModel):
    def __init__(self, models, weights=None):
        CaptionModel.__init__(self)
        # super(AttEnsemble, self).__init__()

        self.models = nn.ModuleList(models)
        self.vocab_size = models[0].vocab_size
        self.seq_length = models[0].seq_length
        self.ss_prob = 0
        weights = weights or [1] * len(self.models)
        self.register_buffer('weights', torch.tensor(weights))

    def init_hidden(self, batch_size):
        return [m.init_hidden(batch_size) for m in self.models]

    def embed(self, it):
        return [m.embed(it) for m in self.models]

    def core(self, *args):
        return zip(*[m.core(*_) for m, _ in zip(self.models, zip(*args))])

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state, tmp_att_masks)
        logprobs = torch.stack([F.softmax(m.logit(output[i]), dim=1) for i,m in enumerate(self.models)], 2).mul(self.weights).div(self.weights.sum()).sum(-1).log()
        return logprobs, state

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = [m.fc_embed(fc_feats) for m in self.models]
        att_feats = [pack_wrapper(m.att_embed, att_feats[...,:m.att_feat_size], att_masks) for m in self.models]

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = [m.ctx2att(att_feats[i]) for i,m in enumerate(self.models)]

        #Replicate the features for l2r and r2l.
        fc_feats = [torch.stack([fc_feats_i, fc_feats_i],dim=1).view(-1, fc_feats_i.size(-1)) for fc_feats_i  in fc_feats]
        att_feats = [torch.stack([att_feats_i, att_feats_i],dim=1).view(-1, att_feats_i.size(-2), att_feats_i.size(-1)) for att_feats_i in att_feats]
        p_att_feats = [torch.stack([p_att_feats_i, p_att_feats_i],dim=1).view(-1, p_att_feats_i.size(-2), p_att_feats_i.size(-1)) for p_att_feats_i in p_att_feats]
        att_masks = torch.stack([att_masks, att_masks],dim=1).view(-1, att_masks.size(-1))


        return fc_feats, att_feats, p_att_feats, [att_masks] * len(self.models)


    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        p_fc_feats = [p_fc_feats_i.view(-1,2,p_fc_feats_i.size(-1))  for p_fc_feats_i  in p_fc_feats]
        p_att_feats = [p_att_feats_i.view(-1,2,p_att_feats_i.size(-2), p_att_feats_i.size(-1)) for  p_att_feats_i  in p_att_feats] 
        pp_att_feats = [pp_att_feats_i.view(-1,2,pp_att_feats_i.size(-2), pp_att_feats_i.size(-1)) for  pp_att_feats_i  in pp_att_feats]
        p_att_masks = [p_att_masks_i.view(-1,2,p_att_masks_i.size(-1)) for p_att_masks_i in p_att_masks] 

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(batch_size,2,self.seq_length).zero_()
        seqLogprobs = torch.FloatTensor(batch_size,2,self.seq_length)
        # lets process every image independently for now, for simplicity
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size*2)
            tmp_fc_feats = [p_fc_feats_i[k:k+1].expand(beam_size, p_fc_feats_i.size(1),p_fc_feats_i.size(2)).contiguous().view(-1, p_fc_feats_i.size(-1)) for p_fc_feats_i  in  p_fc_feats]
            tmp_att_feats = [p_att_feats_i[k:k+1].expand(*((beam_size,)+p_att_feats_i.size()[1:])).contiguous().view(-1, p_att_feats_i.size(-2), p_att_feats_i.size(-1))  for  p_att_feats_i  in  p_att_feats]
            tmp_p_att_feats = [pp_att_feats_i[k:k+1].expand(*((beam_size,)+pp_att_feats_i.size()[1:])).contiguous().view(-1, pp_att_feats_i.size(-2), pp_att_feats_i.size(-1)) for  pp_att_feats_i in  pp_att_feats]
            tmp_att_masks = [p_att_masks_i[k:k+1].expand(*((beam_size,)+p_att_masks_i.size()[1:])).contiguous().view(-1, p_att_masks_i.size(-1)) for  p_att_masks_i in  p_att_masks] if p_att_masks is not None else None

            for t in range(1):
                if t == 0: # input <l2r> and <r2l>
                    it = fc_feats.new_zeros([beam_size, 2], dtype=torch.long)
                    it[:,0] = self.vocab_size -1 
                    it[:,1] = self.vocab_size
                    it = it.view(-1)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)
            
            logprobs = logprobs.view(-1,2,logprobs.size(-1))
            state = [(state_i[0].view(state_i[0].size(0), -1,2, state_i[0].size(-1)), state_i[1].view(state_i[1].size(0), -1,2, state_i[1].size(-1))) for state_i  in state]
            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            seq[k,0,:] = self.done_beams[k][0][0]['seq']
            seq[k,1,:] = self.done_beams[k][1][0]['seq'] 
            seqLogprobs[k,0,:] = self.done_beams[k][0][0]['logps']
            seqLogprobs[k,1,:] = self.done_beams[k][1][0]['logps'] 
        # return the samples and their log likelihoods
        return seq, seqLogprobs



    def beam_search(self, init_state, init_logprobs, *args, **kwargs):

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - diversity_lambda
            return unaug_logprobsf

        # does one step of classical beam search

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam
            ys,ix = torch.sort(logprobsf,1,True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    #compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q,c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    local_unaug_logprob = unaug_logprobsf[q,ix[q,c]]
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_unaug_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            # new_state = [_.clone() for _ in state]
            new_state = [[_.clone() for _ in state_] for state_ in state]
            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                # #rearrange recurrent states
                # for state_ix in range(len(new_state)):
                # #  copy over state in previous beam q to new beam at vix
                #     new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                #rearrange recurrent states
                for ii in range(len(new_state)):
                    for state_ix in range(len(new_state[ii])):
                    #  copy over state in previous beam q to new beam at vix
                        new_state[ii][state_ix][:, vix] = state[ii][state_ix][:, v['q']] # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            state = new_state
            return beam_seq,beam_seq_logprobs,beam_logprobs_sum,state,candidates

        # Start diverse_beam_search
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        max_ppl = opt.get('max_ppl', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size # beam per group
        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(self.seq_length, bdash, 2).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash, 2).zero_() for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash,2) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[[],[]] for _ in range(group_size)]
        state_table = [[list(torch.unbind(_)) for _ in torch.stack(init_state_).chunk(group_size, 2)] for init_state_  in  init_state]
        # state_table =list(zip(*[[list(torch.unbind(_)) for _ in torch.stack(init_state_).chunk(group_size, 2)] for init_state_  in  init_state]))
        logprobs_table = list(init_logprobs.chunk(group_size, 0))
        # END INIT

        # Chunk elements in the args
        args = list(args)
        args = [[_.chunk(group_size) if _ is not None else [None]*group_size for _ in args_] for args_ in args] # arg_name, model_name, group_name
        args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in range(group_size)] # group_name, arg_name, model_name

        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size): 
                if t >= divm and t <= self.seq_length + divm - 1:
                    # add diversity
                    logprobsf = logprobs_table[divm].data.float()
                    # suppress previous word
                    if decoding_constraint and t-divm > 0:
                        logprobsf.scatter_(2, beam_seq_table[divm][t-divm-1].unsqueeze(2).cuda(), float('-inf'))
                    # suppress UNK tokens in the decoding, the last two are <l2r> and <r2l>
                    logprobsf[:,:,logprobsf.size(2)-3] = logprobsf[:,:, logprobsf.size(2)-3] - 1000  
                    # diversity is added here
                    # the function directly modifies the logprobsf values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    unaug_logprobsf = add_diversity(beam_seq_table,logprobsf,t,divm,diversity_lambda,bdash)



                    for i in range(unaug_logprobsf.size(1)):

                        # infer new beams
                        beam_seq_table[divm][:,:,i],\
                        beam_seq_logprobs_table[divm][:,:,i],\
                        beam_logprobs_sum_table[divm][:,i],\
                        state_table_tmp,\
                        candidates_divm = beam_step(logprobsf[:,i,:],
                                                    unaug_logprobsf[:,i,:],
                                                    bdash,
                                                    t-divm,
                                                    beam_seq_table[divm][:,:,i],
                                                    beam_seq_logprobs_table[divm][:,:, i],
                                                    beam_logprobs_sum_table[divm][:,i],
                                                    [[state_table_i[divm][0][:,:,i,:], state_table_i[divm][1][:,:,i,:]] for state_table_i in state_table])
                        # [[state_table_i[divm][0][:,:,i,:], state_table_i[divm][1][:,:,i,:]] for state_table_i in state_table ]
                        for state_table_ix in range(len(state_table)):
                            [state_table[state_table_ix][divm][0][:,:,i,:], state_table[state_table_ix][divm][1][:,:,i,:]] = state_table_tmp[state_table_ix]
                        # if time's up... or if end token is reached then copy beams
                        for vix in range(bdash):
                            if beam_seq_table[divm][t-divm,vix,i] == 0 or t == self.seq_length + divm - 1:
                                final_beam = {
                                    'seq': beam_seq_table[divm][: , vix, i].clone(), 
                                    'logps': beam_seq_logprobs_table[divm][:, vix, i].clone(),
                                    'unaug_p': beam_seq_logprobs_table[divm][:, vix, i].sum().item(),
                                    'p': beam_logprobs_sum_table[divm][vix,i].item()
                                }
                                final_beam['p'] = length_penalty(t-divm+1, final_beam['p'])
                                # if max_ppl:
                                #     final_beam['p'] = final_beam['p'] / (t-divm+1)
                                done_beams_table[divm][i].append(final_beam)
                                # don't continue beams from finished sequences
                                beam_logprobs_sum_table[divm][vix,i] = -1000

                    # move the current group one step forward in time
                    it = beam_seq_table[divm][t-divm].view(-1)
                    state_table_divm = [[state_table_i[divm][0].view(state_table_i[divm][0].size(0),-1,state_table_i[divm][0].size(-1)), state_table_i[divm][1].view(state_table_i[divm][1].size(0),-1,state_table_i[divm][1].size(-1))] for state_table_i in state_table]
                    logprobs_tmp, state_tmp =  self.get_logprobs_state(it.cuda(), *(args[divm] + [state_table_divm]))
                    logprobs_tmp = logprobs_tmp.view(-1,2,logprobs_tmp.size(-1))
                    state_tmp = [(state_tmp_i[0].view(state_tmp_i[0].size(0), -1,2, state_tmp_i[0].size(-1)), state_tmp_i[1].view(state_tmp_i[1].size(0), -1,2, state_tmp_i[1].size(-1))) for state_tmp_i  in state_tmp]
                    logprobs_table[divm] = logprobs_tmp
                    for state_table_ix in range(len(state_table)):
                        [state_table[state_table_ix][divm][0], state_table[state_table_ix][divm][1]] = state_tmp[state_table_ix]

        # all beams are sorted by their log-probabilities
        # done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        # done_beams = reduce(lambda a,b:a+b, done_beams_table)
        done_beams = []
        for i in range(group_size):
            for j in range(2):
                done_beams.append(sorted(done_beams_table[i][j], key=lambda x: -x['p'])[:bdash])
        
        return done_beams