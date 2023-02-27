parser grammar qLispParser;

options {
	tokenVocab = qLispLexer;
}

prog: expr # program;

expr:
	atom																			# atomExpr
	| BEGIN LPAREN body = expr_list RPAREN											# beginExpr
	| IF LPAREN cond = expr COMMA true_case = expr COMMA false_case = expr RPAREN	# ifExpr
	| WHILE LPAREN cond = expr COMMA body = expr RPAREN								# whileExpr
	| head = expr LPAREN (binds = bind_list)? RPAREN								# applyExpr
	| LPAREN (binds = bind_list)? RPAREN MAP body = expr							# lambdaExpr
	| LPAREN body = expr RPAREN														# parenExpr
	| op = (PLUS | MINUS) expr														# unaryExpr
	| left = expr POWER right = expr												# powerExpr
	| left = expr op = (TIMES | DIVIDE | MODULO) right = expr						# mulDivModExpr
	| left = expr op = (PLUS | MINUS) right = expr									# addSubExpr
	| left = expr op = (EQ | NEQ | LT | GT | LTE | GTE) right = expr				# compareExpr
	| NOT expr																		# notExpr
	| left = expr XOR right = expr													# xorExpr
	| left = expr AND right = expr													# andExpr
	| left = expr OR right = expr													# orExpr
	| left = expr (LSHIFT | RSHIFT) right = expr									# shiftExpr
	| left = expr LOGICAL_AND right = expr											# logicalAndExpr
	| left = expr LOGICAL_OR right = expr											# logicalOrExpr
	| define																		# defineExpr
	| assign																		# assignExpr;

expr_list: first = expr COMMA rest = expr_list | first = expr;

define:
	name = IDENTIFIER DEFINE body = expr										# defineConstant
	| name = IDENTIFIER LPAREN (binds = bind_list)? RPAREN DEFINE body = expr	# defineFunction;

assign: name = IDENTIFIER ASSIGN body = expr;

bind:
	value = expr							# positionBind
	| name = IDENTIFIER BIND value = expr	# namedBind;

bind_list: first = bind COMMA rest = bind_list | first = bind;

atom:
	QUBITS			# qubits
	| IDENTIFIER	# identifier
	| INT			# int
	| HEX			# hex
	| OCT			# oct
	| BIN			# bin
	| FLOAT			# float
	| COMPLEX		# complex
	| STRING		# string;
