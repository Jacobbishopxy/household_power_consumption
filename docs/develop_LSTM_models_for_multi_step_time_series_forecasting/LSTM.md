# Understanding LSTM

[link](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### LSTM Networks

Long Short Term Memory networks - usually just called "LSTM" - are a special kind of RNN,
capable of learning long-term dependencies. They work tremendously well on a large variety
of problems, and are now widely used.

LSTM are explicitly designed to avoid the long-term dependency problem. Remembering
information for long periods of time is practically their default behavior, not something
they struggle to learn!

All recurrent neural networks have the form of a chain of repeating modules of neural
network. In standard RNNs, this repeating module will have a very simple structure, such
as a single tanh layer.

![](./img/LSTM3-SimpleRNN.png)

LSTMs also have this chain like structure, but the repeating module has a different 
structure. Instead of having a single neural network layer, there are four, interacting in 
a very special way.

![](./img/LSTM3-chain.png)

![](./img/LSTM2-notation.png)

In the above diagram, each line carries an entire vector, from the output of one node to
the inputs of others. The pink circles represent pointwise operations, like vector addition,
while the yellow boxes are learned neural network layers. Lines merging denote concatenation,
while a line forking denote its content being copied and the copies going to different
locations.


### The Core Idea Behind LSTMs

The key to LSTMs is the cell state, the horizontal line running through the top of the
diagram.
The cell state is kind of like a conveyor belt. It runs straight down the entire chain,
with only some minor linear interactions. It's very easy for information to just flow
along it unchanged.

![image info](./img/LSTM3-C-line.png)

The LSTM does have the ability to remove or add information to the cell state, carefully
regulated by structures called gates.

Gates are a way to optionally let information through. They are composed out of a sigmoid
neural net layer and a pointwise multiplication operation.

![image info](./img/LSTM3-gate.png)

The sigmoid layer outputs numbers between zero and one, describing how much of each 
component should be let through. A value of zero means "let nothing through", while a
value of one means "let everything through!"

An LSTM has three of these gates, to protect and control the cell state.


### Step-by-Step LSTM Walk Through

The first step in out LSTM is to decide what information we're going to throw away from
the cell state. This decision is made by a sigmoid layer called the "forget gate layer".
It looks at h_t-1 and x_t, and outputs a number between 0 and 1 for each number in the
cell state C_t-1. A 1 represents "completely keep this" while a 0 represents "completely
get rid of this".

Let's go back to our example of a language model trying to predict the next word based
on all the previous ones. In such a problem, the cell state might include the gender of
the present subject, so that the correct pronouns can be used. When we see a new subject,
we want to forget the gender of the old subject.

![image info](./img/LSTM3-focus-f.png)

The next step is to decide what new information we're going to store in the cell state.
This has two parts. First, a sigmoid layer called the "input gate layer" decides which 
values we'll update. Next, a tanh layer creates a vector of new candidate values, ~C_t,
that could be added to the state. In the next step, we'll combine these two to create an
update to the state.

In the example of our language model, we'd want to add the gender of the new subject to 
the cell state, to replace the old we're forgetting.

![image info](./img/LSTM3-focus-i.png)

It's now time to update the old cell state, C_t-1, into the new cell state C_t. The
previous steps already decided what to do, we just need to actually do it.

We multiple the old state by f_t, forgetting the things we decided to forget earlier. 
Then we add i_t * ~C_t. This is the new candidate values, scaled by how much we decided
to update each state value.

In the case of the language model, this is where we'd actually drop the information about
the old subject's gender and add the new information , as we decided in the previous steps.

![image info](img/LSTM3-focus-c.png)

Finally, we need to decide what we're going to output. This output will be based on our 
cell state, but will be a filtered version. First, we run a sigmoid layer which decides 
what parts of the cell state we're going to output. Then, we put the cell state through 
tanh (to push the values to be between -1 and 1) and multiply it by the output of the 
sigmoid gate, so that we only output the parts we decided to.

For the language model example, since it just saw a subject, it might want to output
information relevant to a verb, in case that's what is coming next. For example, it might
output whether th subject is singular or plural, so that we know what form a verb should
be conjugated into if that's what follows next.

![image info](./img/LSTM3-focus-o.png)


### Variants on Long Short Term Memory

One popular LSTM variant is adding "peephole connections." This means that we let the gate
layers look at the cell state.

![var peepholes](./img/LSTM3-var-peepholes.png)

The above diagram adds peepholes to all the gates, but many papers will give some peepholes 
and not others.

Another variation is to use coupled forget and input gates. Instead of separately deciding
what to forget and what we should add new information to, we make those decisions together.
We only forget when we're going to input something in its place. We only input new values 
to the state when we forget something older.

![var tied](./img/LSTM3-var-tied.png)

A slightly more dramatic variation on the LSTM is the Gated Recurrent Unit, or GRU. It
combines the forget and input gates into a single "update gate". It also merges the cell
state and hidden state, and makes some other changes. The resulting model is simpler than
standard LSTM models, and has been growing increasingly popular.

![var GRU](./img/LSTM3-var-GRU.png)

These are only a few of the most notable LSTM variants. There are lots of others, like
Depth Gated RNNs. There's also some completely different approach to tacking long-term
dependencies.
