--[[
   Deep Exploration via Bootstrapped DQN
   Ian Osband, Charles Blundell, Alexander Pritzel, Benjamin Van Roy
   Adapted by Rakesh R Menon, Manu S Halvagal
   Usage: nn.Bootstrap(nn.Linear(size_in, size_out), 10, 0.08)
]]--

local Bootstrap, parent = torch.class('nn.Bootstrap', 'nn.Module')

function Bootstrap:__init(mod, k, param_init)
    parent.__init(self)

    self.k = k
    self.active = 1
    self.param_init = param_init or 0.1
    self.mod = mod:clearState()
    self.mods = {}
    self.mods_container = nn.Container()

    for k=1,self.k do
        if self.param_init then
            self.mods[k] = self.mod:clone():reset(self.param_init / math.sqrt(3))
        else    
            self.mods[k] = self.mod:clone():reset()
        end
        self.mods_container:add(self.mods[k])
    end
end

function Bootstrap:clearState()
    self.mods_container:clearState()
    return parent.clearState(self)
end

function Bootstrap:parameters(...)
    return self.mods_container:parameters(...)
end

function Bootstrap:type(type, tensorCache)
    return parent.type(self, type, tensorCache)
end

function Bootstrap:updateOutput(input) 
    if input:dim() == 1 then
        self.output:resize(self.mod.weight:size(1))
    elseif input:dim() == 2 then
        local nframe = input:size(1)
        self.output:resize(nframe, self.mod.weight:size(1))
    end
    self.output:zero()

    for i=1,self.k do
        self.output:add(self.mods[i]:updateOutput(input))
    end
    self.output:div(self.k)
    return self.output
end

function Bootstrap:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input):zero()
    for i=1,self.k do
        self.gradInput:add(self.mods[i]:updateGradInput(input, gradOutput[1]):div(self.k))
    end

    return self.gradInput
end

function Bootstrap:accGradParameters(input, gradOutput, scale)
    for i=1,self.k do
        self.mods[i]:accGradParameters(input, gradOutput, scale):div(self.k)    
    end
end