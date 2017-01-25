local GradScale, parent = torch.class('nn.GradScale', 'nn.Module')

function GradScale:__init(scale)
      parent.__init(self)
      self.scale = 1/scale
end

function GradScale:updateOutput(input)
    self.output:resizeAs(input):copy(input)
    return self.output
end

function GradScale:updateGradInput(input, gradOutput)
      self.gradInput = gradOutput:resizeAs(input):mul(self.scale)
      return self.gradInput
end
function GradScale:set_scale(new_scale)
      self.scale = 1/new_scale
end
