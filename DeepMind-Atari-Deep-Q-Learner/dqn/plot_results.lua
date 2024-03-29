require 'nn'
require 'initenv'
require 'cutorch'
require 'gnuplot'
require 'NeuralQLearner'

if #arg < 1 then
  print('Usage: ', arg[0], ' <DQN file>')
  return
end

data = torch.load(arg[1])
data3 = torch.load(arg[2])

--gnuplot.raw('set multiplot layout 2, 3')

gnuplot.pdffigure('Average_reward.pdf')
gnuplot.title('Average reward per game during testing')
gnuplot.plot({'Double DQN',torch.Tensor(data.reward_history)[{{1,20}}]},{'BootstrapDQN 2 heads',torch.Tensor(data3.reward_history)[{{1,20}}]})
gnuplot.xlabel('Number of Epochs')
gnuplot.ylabel('Average Reward')
gnuplot.plotflush()

gnuplot.pdffigure('Total_count_reward.pdf')
gnuplot.title('Total count of rewards during testing')
gnuplot.plot({'Double DQN',torch.Tensor(data.reward_counts)[{{1,20}}]},{'BootstrapDQN 2 heads',torch.Tensor(data3.reward_counts)[{{1,20}}]})
gnuplot.xlabel('Number of Epochs')
gnuplot.ylabel('Average Reward')
gnuplot.plotflush()

gnuplot.pdffigure('Num_games.pdf')
gnuplot.title('Number of games played during testing')
gnuplot.plot({'Double DQN',torch.Tensor(data.episode_counts)[{{1,20}}]},{'BootstrapDQN 2 heads',torch.Tensor(data3.episode_counts)[{{1,20}}]})
gnuplot.xlabel('Number of Epochs')
gnuplot.ylabel('Number of Games')
gnuplot.plotflush()

gnuplot.pdffigure('Average_q_val.pdf')
gnuplot.title('Average Q-value of validation set')
gnuplot.plot({'Double DQN',torch.Tensor(data.v_history)[{{1,20}}]},{'BootstrapDQN 2 heads',torch.Tensor(data3.v_history)[{{1,20}}]})
gnuplot.xlabel('Number of Epochs')
gnuplot.ylabel('Average Q Value')
gnuplot.plotflush()

gnuplot.pdffigure('TD error.pdf')
gnuplot.title('TD error (old and new Q-value difference) of validation set')
gnuplot.plot({'Double DQN',torch.Tensor(data.td_history)[{{1,20}}]},{'BootstrapDQN 2 heads',torch.Tensor(data3.td_history)[{{1,20}}]})
gnuplot.xlabel('Number of Epochs')
gnuplot.ylabel('TD error')
gnuplot.plotflush()

gnuplot.pdffigure('Seconds.pdf')
gnuplot.title('Seconds elapsed after epoch')
gnuplot.plot({'Double DQN',torch.Tensor(data.time_history)[{{1,20}}]},{'BootstrapDQN 2 heads',torch.Tensor(data3.time_history)[{{1,20}}]})
gnuplot.xlabel('Number of Epochs')
gnuplot.ylabel('Seconds elapsed(s)')
gnuplot.plotflush()

--gnuplot.figure()
--gnuplot.title('Qmax history')
--gnuplot.plot(torch.Tensor(data.qmax_history))

