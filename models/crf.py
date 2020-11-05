import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import List,Optional

class CRF(nn.Module):
    def __init__(self,num_tags: int, batch_first:bool=False) -> None:
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags:{num_tags}")
        super().__init__()
        self.num_tags = num_tags 
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags,num_tags))

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.start_transitions,-0.1,0.1)
        nn.init.uniform_(self.end_transitions,-0.1,0.1)
        nn.init.uniform_(self.transitions,-0.1,0.1)

    def __repr__(slef) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"
    
    def forward(self,emissions: torch.Tensor,
                tags: torch.LongTensor,
                mask: Optional[torch.ByteTensor] = None,
                reduction: str = "mean") -> torch.Tensor:
        """ 
        Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions: (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length,batch_size,num_tags) if batch_first is False
                ``(batch_size,seq_length,num_tags) otherwise
            tags: (~ torch.LongTensor) : Sequence of tags tensor of size
                ``(seq_length,batch_size) if batch_first is False
                ``(batch_size,seq_length) otherwise
            mask: (~torch.ByteTensor) : Mask tensor of size 
                ``(seq_length,batch_size) if batch_first is False
                ``(batch_size,seq_length) otherwise
            reduction: Specifies the reduction to apply to the output:
                ``none|sum|mean|token_mean. "none" : no reduction will be applied
        
        Returns: 
            `~torch.Tensor: the log likelihood. This will have size (batch_size)
            if reduction is none
        """
        if reduction not in ["none","sum","mean","token_mean"]:
            raise ValueError(f"invalid reduction:{reduction}")
        if mask is None:
            mask = torch.ones_like(tags,dtype=torch.uint8,device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions,tags=tags,mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0,1)
            tags = tags.transpose(0,1)
            mask = mask.transpose(0,1)
        
        # shape (batch_size,)
        numerator = self._compute_score(emissions,tags,mask)
        # shape (batch_size,)
        denominator = self._compute_normalizer(emissions,mask)
        # shape (batch_size,)
        llh = numerator - denominator

        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        if reduction == "mean":
            return llh.mean()
        
        return llh.sum()/mask.float().sum()
    
    def decode(self,emissions,mask = None,nbest = None, pad_tag = None):
        """
        find the most likely tag sequence using Viterbi algorithm
        Args:
            emissions: emission score tensor of size (seq_length,batch_size,num_tags)
            mask: mask tensor of size (seq_length,batch_size)
            nbest: number of most porbable paths for each sequence
            pad_tag: tag at padded positions.
        Returns:
            a pytorch tensor of the best tag sequence for each batch of shape
            (nbest,batch_size,seq_length)
        """
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2],dtype=torch.uint8,
                                                device = emissions.device
            )
            mask = mask.byte()
        
        self._validate(emissions,mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0,1)
            mask = mask.transpose(0,1)
        
        if nbest == 1:
            return self._viterbi_decode(emissions,mask,pad_tag).unsqueeze(0)
        
        return self._viterbi_decode_nbest(emissions,mask,nbest,pad_tag)



    

    def _validate(self,emissions,tags=None,mask=None):
        if emissions.dim() != 3:
            raise ValueError(f"emissions must have dimension 3,got {emissions.dim()}")
        if emissions.size(2) != self.num_tags:
            raise ValueError(f"expected last dimension of emissions is {self.num_tags}",
                    f"got {self.emissions.size(2)}")
        
        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError("the first two dimension of emssions and tags must match",
                f"got  {tuple(emissions.shape[:2])} and {tuple(tags.shape)}"
            )
        
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(f"the first two dimension of emssions and tags must match got  {tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
            )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:,0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask of the first timestep must all be on")


    def _compute_score(self,emissions,tags,mask):

        #emissions : (seq_length,batch_size,num_tags)
        #tags : (seq_length,batch_size)

        seq_length, batch_size = tags.shape
        mask = mask.float()

        score = self.start_transitions[tags[0]]
        score += emissions[0,torch.arange(batch_size),tags[0]]

        for seq  in range(1,seq_length):
            score += self.transitions[tags[seq-1],tags[seq]] * mask[seq]

            score += emissions[seq,torch.arange(batch_size),tags[seq]] * mask[seq]

        
        seq_ends = mask.long().sum(dim=0) -1
        last_tags = tags[seq_ends,torch.arange(batch_size)]

        score += self.end_transitions[last_tags]

        return score

        
    def _compute_normalizer(self,emissions,mask):
        seq_length = emissions.size(0)
        
        # (batch_size,num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1,seq_length):
            # (batch_size,num_tags,1)
            boardcast_score = score.unsqueeze(2)

            boardcast_emissions = emissions[i].unsqueeze(1)

            next_score = boardcast_score + boardcast_emissions + self.transitions

            next_score = torch.logsumexp(next_score,dim=1)

            score = torch.where(mask[i].unsqueeze(1),next_score,score)

        score += self.end_transitions

        return torch.logsumexp(score,dim=1)


    def _viterbi_decode(self,emissions,mask,pad_tag):

        if pad_tag is None:
            pad_tag = 0
        
        device = emissions.device
        seq_length ,batch_size = mask.shape

        score = self.start_transitions + emissions[0]

        history_idx = torch.zeros((seq_length,batch_size,self.num_tags),dtype=torch.long,device=device)
        oor_idx = torch.zeros((batch_size,self.num_tags),dtype=torch.long,device=device)
        oor_tag = torch.full((seq_length,batch_size),pad_tag,dtype=torch.long,device=device)

       

        for i in range(1,seq_length):
            
            # batch_size,num_tags,1
            boardcast_score = score.unsqueeze(2)

            # batch_size,1,num_tags
            boardcast_emissions = emissions[i].unsqueeze(1)

            next_score = boardcast_score + self.transitions + boardcast_emissions

            next_score, indices = next_score.max(dim=1)


            score = torch.where(mask[i].unsqueeze(-1),next_score,score)

            indices = torch.where(mask[i].unsqueeze(-1),indices,oor_idx)

            history_idx[i-1] = indices

        end_score = score + self.end_transitions
        _,end_tag = end_score.max(dim=1)

        seq_ends = mask.long().sum(dim=0) -1

        history_idx = history_idx.transpose(1,0).contiguous()
        history_idx.scatter_(1,seq_ends.view(-1,1,1).expand(-1,1,self.num_tags),
                            end_tag.view(-1,1,1).expand(-1,1,self.num_tags)
        )
        history_idx = history_idx.transpose(1,0).contiguous()

        best_tags_arr = torch.zeros((seq_length,batch_size),dtype=torch.long,device=device)
        best_tags = torch.zeros(batch_size,1,dtype=torch.long,device=device)

        for idx in range(seq_length-1,-1,-1):
            best_tags = torch.gather(history_idx[idx],1,best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size)
        return torch.where(mask,best_tags_arr,oor_tag).transpose(0,1)


    def _viterbi_decode_nbest(self,emissions,mask,nbest,pad_tag=None):
        if pad_tag is None:
            pad_tag = 0
        
        device = emissions.device
        seq_length,batch_size = emissions.shape[:2]
        # (batch,num_tags)
        score = self.start_transitions + emissions[0]

        history_idx = torch.zeros((seq_length,batch_size,self.num_tags,nbest),dtype=torch.long,device=device)
        oor_index = torch.zeros((batch_size,self.num_tags,nbest),dtype=torch.long,device=device)
        oor_tag  = torch.full((seq_length,batch_size,nbest),pad_tag,dtype=torch.long,device=device)


        for i in range(1,seq_length):

            if i == 1:
                # (batch_size,num_tag,1)
                boardcast_score = score.unsqueeze(-1)

                # batch_size,1,num_tag

                boardcast_emissions = emissions[i].unsqueeze(1)

                # batch_size,num_tag,num_tag
                next_score = self.transitions + boardcast_score + boardcast_emissions

            else:
                # batch_size,num_tag,nbest,1
                boardcast_score = score.unsqueeze(-1)
                # batch,1,1,num_tag
                boardcast_emissions = emissions[i].unsqueeze(1).unsqueeze(2)

                # batch_size,num_tag,nbest,num_tag
                next_score = self.transitions.unsqueeze(1) + boardcast_score + boardcast_emissions

            
            next_score,indexs = next_score.view(batch_size,-1,self.num_tags).topk(nbest,dim=1)

            if i == 1:
                score = score.unsqueeze(-1).expand(-1,-1,nbest)
                indexs = indexs * nbest

            next_score = next_score.transpose(1,2)
            indexs = indexs.transpose(1,2)

            score = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1),next_score,score)
            indexs = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1),indexs,oor_index)
            history_idx[i-1] = indexs

        
        # end
        end_score = score + self.end_transitions.unsqueeze(-1)
        _,end_tag = end_score.view(batch_size,-1).topk(nbest,dim=1)

        # batch_size,
        seq_ends = mask.long().sum(dim=0) -1

        history_idx = history_idx.transpose(1,0).contiguous()
        history_idx.scatter_(1,seq_ends.view(-1,1,1,1).expand(-1,1,self.num_tags,nbest),
                                end_tag.view(-1,1,1,nbest).expand(-1,1,self.num_tags,nbest)
        )
        # seq_length,batch_size, num_tag,nbest
        history_idx = history_idx.transpose(1,0).contiguous()


        best_tags_arr = torch.zeros((seq_length,batch_size,nbest),dtype=torch.long,device=device)

        best_tags = torch.arange(nbest,dtype=torch.long,device=device).view(1,-1).expand(batch_size,-1)

        for idx in range(seq_length-1,-1,-1):
            best_tags = torch.gather(history_idx[idx].view(batch_size,-1),1,best_tags)

            # 这里处nbest很关键
            best_tags_arr[idx] = best_tags.data.view(batch_size,-1) // nbest

        return torch.where(mask.unsqueeze(-1),best_tags_arr,oor_tag).permute(2,1,0)



if __name__ == "__main__":
    num_tags = 5
    batch_size = 8
    seq_length = 12

    emissions = (torch.randn((seq_length,batch_size,num_tags),dtype=torch.float))
    tags = (torch.randint(0,5,(seq_length,batch_size),dtype=torch.long))
    print(f"emissions : {emissions},{emissions.size()}")
    # print(f"tags: {tags}")
    crf = CRF(num_tags)
    
    loss = crf(emissions,tags)
    decode = crf.decode(emissions,None)
    print("decode is ",decode)


            





            











    

           





            








    
        

            


    