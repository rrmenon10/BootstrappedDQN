--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "initenv"
require "nn"
require "Bootstrap"

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

    -- THIS PART FOR NORMAL DQN 


    -- fully connected layer
    net:add(nn.Linear(nel, args.n_hid[1]))
    net:add(args.nl())
    -- local last_layer_size = args.n_hid[1]

    -- for i=1,(#args.n_hid-1) do
    --     add Linear layer
    --     last_layer_size = args.n_hid[i+1]
    --     net:add(nn.Linear(args.n_hid[i], last_layer_size))
    --     net:add(args.nl())
    -- end

    -- add the last fully connected layer (to actions)
    -- net:add(nn.Linear(last_layer_size, args.n_actions))

    -- THIS PART FOR BOOTSTRAP DQN

    net:add(nn.Bootstrap(nn.Linear(args.n_hid[1],args.n_actions),10,0.08))

    -- THIS PART FOR SOFT ATTENTION
    -- head_att = nn.ConcatTable()
    -- heads = nn.ConcatTable()
    -- for i=1,10 do
    --     mlp = nn.Sequential()
    --     mlp:add(nn.Linear(args.n_hid[1],args.n_actions))
    --     mlp:add(args.nl())
    --     -- for j=1,(#args.n_hid-1) do
    --     --     mlp:add(nn.Linear(args.n_hid[i], args.n_hid[i+1]))
    --     --     mlp:add(args.nl())
    --     -- end
    --     -- mlp:add(nn.Linear(last_layer_size, args.n_actions))
    --     heads:add(mlp)
    -- end
    -- att = nn.Sequential()
    -- att:add(nn.Linear(args.n_hid[1],64))
    -- att:add(nn.Linear(64,10))
    -- att:add(nn.SoftMax())
    -- head_att:add(heads)
    -- head_att:add(att)
    -- net:add(head_att)

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    return net
end
