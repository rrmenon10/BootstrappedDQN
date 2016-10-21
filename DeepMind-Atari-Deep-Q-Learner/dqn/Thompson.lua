--[[
   Deep Exploration via Thompson DQN
   Ian Osband, Charles Blundell, Alexander Pritzel, Benjamin Van Roy
   Implemented by Yannis M. Assael (www.yannisassael.com), 2016
   Usage: nn.Thompson(nn.Linear(size_in, size_out), 10, 0.08)
]]--

local Thompson, parent = torch.class('nn.Thompson', 'nn.Module')

function Thompson:__init(mod, k, param_init)
    parent.__init(self)
    
    self.k = k
    self.active = {}
    self.param_init = param_init
    self.mod = mod:clearState()
    self.mods = {}
    self.mods_container = nn.Container()

    for k=1,self.k do
        if self.param_init then
            -- By default nn.Linear multiplies with math.sqrt(3)
            self.mods[k] = self.mod:clone():reset(self.param_init / math.sqrt(3))
        else    
            self.mods[k] = self.mod:clone():reset()
        end
        self.mods_container:add(self.mods[k])
    end
end

function Thompson:clearState()
    self.mods_container:clearState()
    return parent.clearState(self)
end

function Thompson:parameters(...)
    return self.mods_container:parameters(...)
end

function Thompson:type(type, tensorCache)
    return parent.type(self, type, tensorCache)
end

function Thompson:updateOutput(input)
    -- resize output    
    if input:dim() == 1 then
        self.output:resize(self.mod.weight:size(1))
    elseif input:dim() == 2 then
        local nframe = input:size(1)
        self.output:resize(nframe, self.mod.weight:size(1))
    end
    self.output:zero()

    -- reset active heads
    self.active = {}

    -- pick a random k
    local k = torch.random(self.k)

    -- select active heads
    for i=1,k do
        self.active[i] = torch.random(self.k)
        self.output:add(self.mods[self.active[i]]:updateOutput(input))
    end
    self.output:div(#self.active)

    return self.output
end

function Thompson:updateGradInput(input, gradOutput)
    -- rescale gradients
    gradOutput:div(#self.active)

    -- resize gradinput
    self.gradInput:resizeAs(input):zero()

    -- accumulate gradinputs
    for i=1,#self.active do
        self.gradInput:add(self.mods[self.active[i]]:updateGradInput(input, gradOutput))
    end

    return self.gradInput
end

function Thompson:accGradParameters(input, gradOutput, scale)
    -- rescale gradients
    gradOutput:div(#self.active)

    -- accumulate grad parameters
    for i=1,#self.active do
        self.mods[self.active[i]]:accGradParameters(input, gradOutput, scale)    
    end
end