import gym
from deep_rl import *

params = {
    "env": "CartPole-v0",
    "seed": 100,
    "max_steps": 400000,
    "action_dim": 2,
    "num_workers": 1,
    "num_options": 2,
}




# set seed
random.seed(params["seed"])
os.environ['PYTHONHASHSEED'] = str(params["seed"])
np.random.seed(params["seed"])
torch.manual_seed(params["seed"])

env = gym.make(params["env"])
env.seed(params["seed"])
env = Task2([env])

config = Config()
config.action_dim = 2
config.state_dim = 4
config.num_workers = params["num_workers"]
config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
config.network_fn = lambda: OptionCriticNet(FCBody(config.state_dim), config.action_dim, num_options=params["num_options"])
config.random_option_prob = LinearSchedule(1.0, 0.1, 1e4)
config.discount = 0.99
config.target_network_update_freq = 20
config.rollout_length = 5
config.termination_regularizer = 0.01
config.entropy_weight = 0.01
config.gradient_clip = 5
agent = OptionCriticAgent2(config)

agent.states = config.state_normalizer(env.reset())

steps = 0
sum_rewards = 0

while steps <= params["max_steps"]:

    storage = Storage(config.rollout_length, ['beta', 'option', 'beta_advantage', 'prev_option', 'init_state', 'eps'])
    for _ in range(config.rollout_length):
        prediction = agent.network(agent.states)
        epsilon = config.random_option_prob(config.num_workers)
        options = agent.sample_option(prediction, epsilon, agent.prev_options, agent.is_initial_states)
        prediction['pi'] = prediction['pi'][agent.worker_index, options]
        prediction['log_pi'] = prediction['log_pi'][agent.worker_index, options]
        dist = torch.distributions.Categorical(probs=prediction['pi'])
        actions = dist.sample()
        entropy = dist.entropy()

        s_, rewards, terminals, info = env.step(actions.cpu().detach().numpy())
        s_ = config.state_normalizer(s_)
        rewards = config.reward_normalizer(rewards)

        storage.feed(prediction)
        storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                      'mask': tensor(1 - terminals).unsqueeze(-1),
                      'option': options.unsqueeze(-1),
                      'prev_option': agent.prev_options.unsqueeze(-1),
                      'entropy': entropy.unsqueeze(-1),
                      'action': actions.unsqueeze(-1),
                      'init_state': agent.is_initial_states.unsqueeze(-1).float(),
                      'eps': epsilon})

        if not terminals[0]:
            sum_rewards += rewards[0]
        else:
            sum_rewards += rewards[0]
            print("Episode reward: {}, num_steps: {}".format(sum_rewards, steps))
            sum_rewards = 0

        agent.is_initial_states = tensor(terminals).byte()
        agent.prev_options = options
        agent.states = s_

        steps += 1
        if steps // config.num_workers % config.target_network_update_freq == 0:
            agent.target_network.load_state_dict(agent.network.state_dict())

    with torch.no_grad():
        prediction = agent.target_network(agent.states)
        storage.placeholder()
        betas = prediction['beta'][agent.worker_index, agent.prev_options]
        ret = (1 - betas) * prediction['q'][agent.worker_index, agent.prev_options] + \
              betas * torch.max(prediction['q'], dim=-1)[0]
        ret = ret.unsqueeze(-1)

    for i in reversed(range(config.rollout_length)):
        ret = storage.reward[i] + config.discount * storage.mask[i] * ret
        adv = ret - storage.q[i].gather(1, storage.option[i])
        storage.ret[i] = ret
        storage.advantage[i] = adv

        v = storage.q[i].max(dim=-1, keepdim=True)[0] * (1 - storage.eps[i]) + storage.q[i].mean(-1).unsqueeze(-1) * \
            storage.eps[i]
        q = storage.q[i].gather(1, storage.prev_option[i])
        storage.beta_advantage[i] = q - v + config.termination_regularizer

    entries = storage.extract(
        ['q', 'beta', 'log_pi', 'ret', 'advantage', 'beta_advantage', 'entropy', 'option', 'action', 'init_state',
         'prev_option'])

    q_loss = (entries.q.gather(1, entries.option) - entries.ret.detach()).pow(2).mul(0.5).mean()
    pi_loss = -(entries.log_pi.gather(1,
                                      entries.action) * entries.advantage.detach()) - config.entropy_weight * entries.entropy
    pi_loss = pi_loss.mean()
    beta_loss = (entries.beta.gather(1, entries.prev_option) * entries.beta_advantage.detach() * (
                1 - entries.init_state)).mean()

    agent.optimizer.zero_grad()
    (pi_loss + q_loss + beta_loss).backward()
    nn.utils.clip_grad_norm_(agent.network.parameters(), config.gradient_clip)
    agent.optimizer.step()