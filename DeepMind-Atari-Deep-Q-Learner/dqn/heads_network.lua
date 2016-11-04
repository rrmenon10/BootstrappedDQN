--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "initenv"
require "nn"
--require "GradScale"
--require "Bootstrap"

return function(args)
    heads = nn.Concat(2)
    --heads:add(nn.GradScale(0.1))
    for i=1,args.num_heads do
        mlp = nn.Sequential()
        mlp:add(nn.Linear(args.n_hid[1],args.n_actions))
        -- mlp:add(args.nl())
        -- for j=1,(#args.n_hid-1) do
        --     mlp:add(nn.Linear(args.n_hid[i], args.n_hid[i+1]))
        --     mlp:add(args.nl())
        -- end
        -- mlp:add(nn.Linear(last_layer_size, args.n_actions))
        heads:add(mlp)
    end

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print("Heads Network Structure")
        print(heads)
        print('Input flattened size:', args.n_hid[1])
    end
    return heads
end
