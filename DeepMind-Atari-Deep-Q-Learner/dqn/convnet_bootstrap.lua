--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "initenv"
require "nn"
require "Replicater"
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

    -- fully connected layer
    net:add(nn.Linear(nel, args.n_hid[1]))
    net:add(args.nl())

    net_sub = nn.Sequential()

        net_sub:add(nn.Replicate(2))
        net_sub:add(nn.SplitTable(1))

        net_subset = nn.ParallelTable()

            -- THIS PART FOR BOOTSTRAPPED DQN

            local bootstrap_headers = nn.Sequential()
            bootstrap_headers:add(nn.Replicater(args.num_heads, args.num_heads))
            bootstrap_headers:add(nn.SplitTable(1))
            -- bootstrap_headers:add(nn.GradScale(1.0/(args.num_heads)))
            local bootstrap_headsubset = nn.ParallelTable()
            for i=1,args.num_heads do
        	   head = nn.Sequential()
        	   head:add(nn.Linear(args.n_hid[1],args.n_actions))
        	   bootstrap_headsubset:add(head)
            end
            bootstrap_headers:add(bootstrap_headsubset)
            net_subset:add(bootstrap_headers)	

            -- THIS PART FOR ATTENTION

            local bootstrap_attention = nn.Sequential()
            

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    return net
end
