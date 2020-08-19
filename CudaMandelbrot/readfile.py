def tokenize(string, token):
	tokenized = ""
	for c in string:
		if c == token:
			return tokenized
		tokenized += c
	return string


def load_values(file):
	f = open(file, 'r')
	lines = [line for line in f]
	values = []

	for line in lines:
		new = line.split(",")
		v = []
		for e in new:
			v.append(int(e))
		values.append(v)

	return values
