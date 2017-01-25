--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "initenv"
require "nn"
require "GradScale"

function create_network(args)

    local net = nn.Sequential()
    net:add(nn.Reshape(unpack(args.input_dims)))

    --- first convolutional layer
    local convLayer = nn.SpatialConvolution

    net:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],1))
    net:add(args.nl())

    -- Add convolutional layers
    for i=1,(#args.n_units-1) do
        -- second convolutional layer
        net:add(convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1]))
        net:add(args.nl())
    end

    local nel
    if args.gpu >= 0 then
        nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims))
                :cuda()):nElement()
    else
        nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
    end

    -- reshape all feature planes into a vector per example
    net:add(nn.Reshape(nel))
    net:add(nn.GradScale(2))

    -- fully connected layer
    -- net:add(nn.Linear(nel, args.n_hid[1]))
    -- net:add(args.nl())

    -- THIS PART FOR BOOTSTRAPPED DQN

    	local bootstrap_headsubset = nn.ConcatTable()
    		for i=1,2 do
	   		head = nn.Sequential()
	   		head:add(nn.Linear(nel, args.n_hid[1]))
	   		head:add(args.nl())
	   		head:add(nn.Linear(args.n_hid[1],args.n_actions))
	   		bootstrap_headsubset:add(head)
    		end
        net:add(bootstrap_headsubset)	

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    return net
end
