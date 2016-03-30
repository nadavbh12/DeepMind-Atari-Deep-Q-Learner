--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

if not dqn then
    require "initenv"
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')
cmd:option('-core', '', 'name of core to use (atari/snes)')
cmd:option('-core_path', '', 'path to core library')
cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-gif_file', '', 'GIF path to write session screens')
cmd:option('-csv_file', '', 'CSV path to write session data')
cmd:option('-display_screen', 0,'if 1 display image from ALE')

cmd:text()

local opt = cmd:parse(arg)

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)
--print ("SETUP completed")
-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end


print("Started playing...")

-- play 30 episodes (game)
for i=1,30 do
  -- start a new game
  local screen, reward, terminal = game_env:newGame()
  episode_reward = 0
  local step = 0
  
  --limit game duration to 5 min
	while not terminal or step < 300 do
	    -- if action was chosen randomly, Q-value is 0
	    agent.bestq = 0
	    
	    -- choose the best action
	    local action_index = agent:perceive(reward, screen, terminal, true, 0.05)

	    -- play game in test mode (episodes don't end when losing a life)
	    screen, reward, terminal = game_env:step(game_actions[action_index], false)
	    
	    --sum new reward
	    episode_reward = episode_reward + reward
	    
	    step = step + 1
	end
	print("reward for episode " .. i .. ": " .. episode_reward)
end

print("Finished playing, close window to exit!")
