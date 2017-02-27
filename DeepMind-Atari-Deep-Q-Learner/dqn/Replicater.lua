local Replicater, parent = torch.class('nn.Replicater','nn.Module')

function Replicater:__init(nf, heads, dim, ndim)
   parent.__init(self)
   self.nfeatures = nf
   self.dim = dim or 1
   self.ndim = ndim
   self.heads = heads
   self.scale = heads
   assert(self.dim > 0, "Can only replicate across positive integer dimensions.")
end

function Replicater:updateOutput(input)
   self.dim = self.dim or 1 --backwards compatible
   assert(
      self.dim <= input:dim()+1,
      "Not enough input dimensions to replicate along dimension " ..
      tostring(self.dim) .. ".")
   local batchOffset = self.ndim and input:dim() > self.ndim and 1 or 0
   local rdim = self.dim + batchOffset
   local sz = torch.LongStorage(input:dim()+1)
   sz[rdim] = self.nfeatures
   for i = 1,input:dim() do
      local offset = 0
      if i >= rdim then
         offset = 1
      end
      sz[i+offset] = input:size(i)
   end
   local st = torch.LongStorage(input:dim()+1)
   st[rdim] = 0
   for i = 1,input:dim() do
      local offset = 0
      if i >= rdim then
         offset = 1
      end
      st[i+offset] = input:stride(i)
   end
   self.output:set(input:storage(),input:storageOffset(),sz,st)
   return self.output
end

function Replicater:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   local batchOffset = self.ndim and input:dim() > self.ndim and 1 or 0
   local rdim = self.dim + batchOffset
   local sz = torch.LongStorage(input:dim()+1)
   sz[rdim] = 1
   for i = 1,input:dim() do
      local offset = 0
      if i >= rdim then
         offset = 1
      end
      sz[i+offset] = input:size(i)
   end
   local gradInput = self.gradInput:view(sz)
   gradInput:sum(gradOutput, rdim):div(self.scale)
   return self.gradInput
end

function Replicater:set_scale(new_scale)
   self.scale = new_scale
end
