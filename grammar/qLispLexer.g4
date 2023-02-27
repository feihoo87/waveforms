lexer grammar qLispLexer;

COMMENT: '//' .*? '\r'? '\n' -> skip;

WS: ( ' ' | '\t' | '\r' | '\n') -> skip;

fragment DIGIT: '0' .. '9';
fragment ALPHA: 'a' .. 'z' | 'A' .. 'Z';
fragment SYMBOL: ('_' | '?' | '!' | '#' | '$' | '%' | '.');

fragment HEX_DIGIT: ( DIGIT | 'a' .. 'f' | 'A' .. 'F');
fragment EXPONENT: ( 'e' | 'E') ( '+' | '-')? DIGIT+;

FLOAT:
	DIGIT+ '.' DIGIT* EXPONENT?
	| '.' DIGIT+ EXPONENT?
	| DIGIT+ EXPONENT;

COMPLEX: (
		DIGIT+ '.' DIGIT* EXPONENT?
		| '.' DIGIT+ EXPONENT?
		| DIGIT+ EXPONENT
	) 'j'
	| DIGIT+ 'j';

INT: DIGIT+;

HEX: '0' ( 'x' | 'X') HEX_DIGIT+;

OCT: '0' ( 'o' | 'O') ( '0' .. '7')+;

BIN: '0' ( 'b' | 'B') ( '0' | '1')+;

STRING: '"' ( '\\"' | ~ ( '"' | '\r' | '\n'))* '"';

IDENTIFIER: (SYMBOL | ALPHA) (SYMBOL | ALPHA | DIGIT)*;
QUBITS: '@' ALPHA ( ALPHA | DIGIT)*;

LPAREN: '(';
RPAREN: ')';
LBRACKET: '[';
RBRACKET: ']';
LBRACE: '{';
RBRACE: '}';
COMMA: ',';
QUOTE: '\'';
BACKQUOTE: '`';
ASSIGN: '=';
BIND: '->';
DEFINE: ':=';
MAP: '=>';
PLUS: '+';
MINUS: '-';
TIMES: '*';
DIVIDE: '/';
MODULO: '%';
POWER: '**';
AND: '&';
OR: '|';
XOR: '^';
LSHIFT: '<<';
RSHIFT: '>>';
LOGICAL_AND: 'and';
LOGICAL_OR: 'or';
NOT: '~';
EQ: '==';
NEQ: '!=';
LT: '<';
LTE: '<=';
GT: '>';
GTE: '>=';
WHILE: 'while';
IF: 'if';
BEGIN: 'begin';
