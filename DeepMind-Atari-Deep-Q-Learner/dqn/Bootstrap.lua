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
    self.active = {}
    self.active[1] = torch.random(self.k)
    self.param_init = param_init or 0.1
    self.mod = mod:clearState()
    self.mods = {}
    self.mods_container = nn.Container()

    for k=1,self.k do
        if self.param_init then
            -- By default nn.Linear multiplies with math.sqrt(3)
            self.mods[k] = self.mod:clone()
            self.mods[k]:reset(self.param_init / math.sqrt(3))
        else    
            self.mods[k] = self.mod:clone()
            self.mods[k]:reset()
        end
        self.mods_container:add(self.mods[k])
    end
end

function Bootstrap:clearState()
    self.active = {}
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
    -- resize output    
    if input:dim() == 1 then
        self.output:resize(self.mod.weight:size(1))
    elseif input:dim() == 2 then
        local nframe = input:size(1)
        self.output:resize(nframe, self.mod.weight:size(1))
    end
    self.output:zero()
    
    local testing = torch.load('test.dat')
    local terminal = torch.load('terminal.dat')
    local i=1
    -- pick a random k
    if testing=="true" then
        self.output = self.mods[1]:updateOutput(input):clone()
        for i=2,self.k do
            self.output:add(self.mods[i]:updateOutput(input))
        end
        self.output:div(self.k)
    else
        if terminal=="true" then
            -- reset active heads
            self.active = {}
            self.active[i] = torch.random(self.k)
        end
        -- print(self.active[i])
        self.output = self.mods[self.active[i]]:updateOutput(input):clone()
        self.output:div(#self.active)
    end
    return self.output
end

function Bootstrap:updateGradInput(input, gradOutput)
    -- rescale gradients
    -- gradOutput:div(self.k)

    -- resize gradinput
    -- self.gradInput:resizeAs(input):zero()

    -- accumulate gradinputs
    self.gradInput = self.mods[1]:updateGradInput(input, gradOutput):clone()
    for i=2,self.k do
        self.gradInput:add(self.mods[i]:updateGradInput(input, gradOutput))
    end
    self.gradInput:div(self.k)

    return self.gradInput
end

function Bootstrap:accGradParameters(input, gradOutput, scale)
    -- rescale gradients
    -- gradOutput:div(self.k)

    -- accumulate grad parameters
    for i=1,self.k do
        self.mods[i]:accGradParameters(input, gradOutput, scale)    
    end
end