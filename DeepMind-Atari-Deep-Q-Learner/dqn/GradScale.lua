local GradScale, parent = torch.class('nn.GradScale', 'nn.Module')

function GradScale:__init(scale)
	parent.__init(self)
	self.scale = scale
end
       
function GradScale:updateOutput(input)
	return input
end

function GradScale:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput:resizeAs(input):mul(self.scale)
	return self.gradInput
end

function GradScale:set_scale(new_scale)
	self.scale = new_scale
end
