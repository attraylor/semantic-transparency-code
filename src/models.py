import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np
import torch.autograd as autograd
import math

def invert_permutation(permutation):
    return np.arange(len(permutation))[np.argsort(permutation)]

def simple_elementwise_apply(fn, packed_sequence):
    """applies a pointwise function fn to each element in packed_sequence"""
    return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)


class RNN(nn.Module):
	def __init__(self, batch_size, output_size, hidden_dim, vocab_size,
				 embedding_length, weights):
		super(RNN, self).__init__()

		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length

		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		if weights is not None:
			self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
		self.rnn = nn.RNN(embedding_length, hidden_dim, num_layers=2,
						  bidirectional=True)
		self.sm_fc = nn.Linear(4*hidden_dim, output_size)

	def forward(self, input_sentences, batch_size=None):

		input = self.word_embeddings(input_sentences)
		input = input.permute(1, 0, 2)
		if batch_size is None:
			if torch.cuda.is_available():
				h_0 = Variable(torch.zeros(4, self.batch_size,
										   self.hidden_dim).cuda()) # 4 = num_layers*num_directions
			else:
				h_0 = Variable(torch.zeros(4, self.batch_size,
									   self.hidden_dim))
		else:
			if torch.cuda.is_available():
				h_0 =  Variable(torch.zeros(4, batch_size,
											self.hidden_dim).cuda())
			else:
				h_0 =  Variable(torch.zeros(4, batch_size, self.hidden_dim))

		output, h_n = self.rnn(input, h_0)
		h_n = h_n.permute(1, 0, 2) # h_n.size() = (batch_size, 4, hidden_dim)
		h_n = h_n.contiguous().view(h_n.size()[0], h_n.size()[1]*h_n.size()[2])
		logits = self.sm_fc(h_n) # logits.size() = (batch_size, output_size)

		return logits

class EntailmentModule(nn.Module):

	def __init__(self, sequence_model, input_dim_size, hidden_dim=32, output_size = 2,
	 			dropout = 0.0, device="cpu", mlp_num_layers = 2, ph_as_one = False):
		#num_layers refers to number of hidden layers. num_layers 3 = input h1 h2 h3 output.
		super(EntailmentModule, self).__init__()
		self.sequence_model = sequence_model
		self.sequence_model.train()
		self.name = "mlp"
		self.input_dim_size = input_dim_size #dimension of premise and hypothesis
		if ph_as_one == False:
			self.num_layers = mlp_num_layers
			self.hidden_dim = hidden_dim
			self.relu = nn.ReLU()
			self.dropout = nn.Dropout(p=dropout)
			self.feedforward1 = nn.Linear(2 * input_dim_size, hidden_dim)
			self.feedforward1.requires_grad = True
			if self.num_layers == 1:
				last_linear_size = 2 * input_dim_size

			if self.num_layers > 1:
				last_linear_size = hidden_dim
				self.sigmoid = nn.Sigmoid()
				self.feedforward2 = nn.Linear(hidden_dim, hidden_dim)
				self.feedforward2.requires_grad = True
			if self.num_layers > 2:
				self.feedforward3 = nn.Linear(hidden_dim, hidden_dim)
				self.feedforward3.requires_grad = True
		else:
			last_linear_size = input_dim_size
		self.last_linear = nn.Linear(last_linear_size, output_size)
		self.last_linear.requires_grad = True
		self.init_weights()

		#self.softmax = nn.LogSoftmax()

		#self.dropout_layer = nn.Dropout(p=dropout)
		self.device = device

	def init_weights(self):
		initrange = 0.1
		self.last_linear.bias.data.zero_()
		self.last_linear.weight.data.uniform_(-initrange, initrange)

	def forward(self, p, pl, h, hl):
		p = self.sequence_model(p, pl)
		if h is not None:
			h = self.sequence_model(h, hl)
			#p, _ = pad_packed_sequence(p, batch_first = True)
			#p, _ = pad_packed_sequence(p, batch_first = True)
			#p, h = (b,d)
			concat = torch.cat((p, h), dim=1) #concat = (b, d*2)
			if self.num_layers > 1:
				out = self.feedforward1(concat)
				if self.num_layers > 1:
					out = self.relu(out)
					out = self.dropout(out)
					out = self.feedforward2(out)
				if self.num_layers > 2:
					out = self.relu(out)
					out = self.dropout(out)
					out = self.feedforward3(out)
			else:
				out = concat
		else:
			out = p
		#out = self.sigmoid(out)
		final_out = self.last_linear(out)

		return final_out

class CBOWClassifier(nn.Module):
	def __init__(self, output_size, vocab_size,
				 embedding_length):
		super(CBOWClassifier, self).__init__()
		self.output_size = output_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		self.hidden_dim = embedding_length
		self.pad_idx = 0
		if torch.cuda.is_available():
			self.device = "cuda"
		else:
			self.device = "cpu"
		self.name = "cbow"

		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
		#self.softmax_fc = nn.Linear(embedding_length, vocab_size)

	def forward(self, input_sentence, lengths=None):
		self.pad_idx = 1
		pad_mask = (input_sentence!=self.pad_idx).to(self.device)
		output = self.word_embeddings(input_sentence)
		output =  output * pad_mask.unsqueeze(2).float()
		output = torch.sum(output, dim=1)
		output = output / lengths.float().unsqueeze(1).to(self.device)
		#output = self.softmax_fc(output)
		return output


class LSTMClassifier(nn.Module):
	def __init__(self, output_size, hidden_dim, vocab_size,
				 embedding_length, num_stacked_lstm = 1, dropout=0.0,pretraining=False,
				 word_embeddings=None, train_word_embeddings = True):
		super(LSTMClassifier, self).__init__()
		self.output_size = output_size
		self.hidden_dim = hidden_dim
		self.bidirectional = False
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		if torch.cuda.is_available():
			self.device = "cuda"
		else:
			self.device = "cpu"
		self.name = "lstm"
		self.pretraining = pretraining

		self.num_layers = num_stacked_lstm

		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
		if word_embeddings is not None:
			self.word_embeddings.weight = nn.Parameter(word_embeddings, requires_grad=train_word_embeddings)
		self.lstm = nn.LSTM(embedding_length, hidden_dim, num_layers=self.num_layers,
						bidirectional=self.bidirectional,
						dropout=dropout,batch_first=True)
		self.softmax_fc = nn.Linear(hidden_dim, vocab_size)


	def init_hidden(self, batch_size):
		if self.bidirectional == True:
			num_directions = 2
		else:
			num_directions = 1
		return(autograd.Variable(torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim).to(self.device)),
					autograd.Variable(torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim).to(self.device)))


	def forward(self, input_sentence, lengths=None):
		self.hidden = self.init_hidden(input_sentence.shape[0])
		lengths, perm_idx = lengths.sort(0, descending=True)
		perm2 = invert_permutation(perm_idx)
		input_sentence = input_sentence[perm_idx]
		packed_sequence = pack_padded_sequence(input_sentence, lengths.cpu().numpy(), batch_first=True)
		input = simple_elementwise_apply(self.word_embeddings, packed_sequence)
		#input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		#input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)

		output, (final_hidden_state, final_cell_state) = self.lstm(input,self.hidden)
		if self.pretraining == True:
			lstm_out, _ = pad_packed_sequence(output, batch_first = True, total_length = input_sentence.shape[1])
			lstm_out = lstm_out[perm2]
			logits = self.softmax_fc(lstm_out.reshape(-1, self.hidden_dim))
			logits_reshaped = logits.reshape(input_sentence.shape[0], input_sentence.shape[1], self.vocab_size)
			return logits_reshaped
		else:
			last_layer = final_hidden_state[-1]
			#final_output = self.sm_fc(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_dim) & final_output.size() = (batch_size, output_size)
			return last_layer[perm2]


# index: M
def batched_index_select(input, dim, index):
	r = torch.zeros(input.shape[0], input.shape[2])
	for i in range(input.shape[0]):
		r[i] = input[i, index[i], :]
	#print(index)
	#print(input.shape)
	return r


class TransformerModel(nn.Module):

	def __init__(self, vocab_size, embedding_dim, nhead=2, hidden_dim=128, nlayers=2, dropout=0.0,
					pretraining=False, word_embeddings=None, train_word_embeddings=True):
		super(TransformerModel, self).__init__()
		from torch.nn import TransformerEncoder, TransformerEncoderLayer
		self.name = 'transformer'
		self.vocab_size = vocab_size
		self.src_mask = None
		self.pad_idx = 0
		self.hidden_dim = hidden_dim
		self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
		encoder_layers = TransformerEncoderLayer(embedding_dim, nhead, hidden_dim, dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
		self.encoder = nn.Embedding(vocab_size, embedding_dim)
		if word_embeddings is not None:
			self.encoder.weight = nn.Parameter(word_embeddings, requires_grad=train_word_embeddings)
		self.embedding_dim = embedding_dim #number of features in input layer
		self.decoder = nn.Linear(embedding_dim, vocab_size)
		self.pretraining = pretraining

		self.init_weights(stoi)
		if torch.cuda.is_available():
			self.device = "cuda"
		else:
			self.device = "cpu"


	def _generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	def init_weights(self, stoi):
		initrange = 0.1
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)
		self.encoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, src, lengths):
		b, l = src.shape
		pad_mask = (src==self.pad_idx).to(self.device)
		src = src.permute(1,0)
		if self.src_mask is None or self.src_mask.size(0) != len(src):
			mask = self._generate_square_subsequent_mask(len(src))
			if torch.cuda.is_available():
				mask = mask.cuda()
			self.src_mask = mask
		src = self.encoder(src) * math.sqrt(self.embedding_dim)
		#print("e", src)
		src = self.pos_encoder(src)
		#print("f", src)
		output = self.transformer_encoder(src, self.src_mask)#, src_key_padding_mask=pad_mask)
		#print("output", output)
		output = output.permute(1,0,2)
		#output = output[perm2]
		#print(output.shape)
		if self.pretraining:
			output = self.decoder(output)
			#output = output.view(-1, self.vocab_size)
			#output = output.reshape(b, l, self.vocab_size)
			return output#, None
		else:
			#output = torch.index_select(output, 1, lens)
			output = batched_index_select(output, 1, lengths - 1).to(self.device)
			#output = output[:,-1,:]
			return output

class PositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=250):
		#d_model: Should be embedding_dim
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)
