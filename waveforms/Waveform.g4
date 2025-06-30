grammar Waveform;

// 主规则
expr: assignment | expression ;

assignment: ID '=' expression ;

expression
    : expression op=('**' | '^') expression                    # PowerExpression
    | expression op=('*' | '/') expression                     # MultiplyDivideExpression
    | expression op=('+' | '-') expression                     # AddSubtractExpression
    | expression op=('<<' | '>>') expression                   # ShiftExpression
    | '(' expression ')'                                       # ParenthesesExpression
    | '-' expression                                           # UnaryMinusExpression
    | functionCall                                             # FunctionCallExpression
    | CONSTANT                                                 # ConstantExpression
    | NUMBER                                                   # NumberExpression
    | STRING                                                   # StringExpression
    | list                                                     # ListExpression
    | tuple                                                    # TupleExpression
    | ID                                                       # IdentifierExpression
    ;

functionCall
    : ID '(' ')'                                               # NoArgFunction
    | ID '(' args ')'                                          # ArgsFunction  
    | ID '(' kwargs ')'                                        # KwargsFunction
    | ID '(' args ',' kwargs ')'                               # ArgsKwargsFunction
    ;

args: expression (',' expression)* ;

kwargs: kwarg (',' kwarg)* ;

kwarg: ID '=' expression ;

list: '[' ']' | '[' expression (',' expression)* ']' ;

tuple: '(' expression ',' ')' | '(' expression (',' expression)+ ')' ;

// 词法规则
NUMBER
    : REAL
    | INT
    | IMAG
    ;

REAL: (DIGIT+ ('.' DIGIT*)? | '.' DIGIT+) ([eE] [+-]? DIGIT+)? ;

INT: DIGIT+ ;

IMAG: (REAL | INT) 'j' ;

STRING: '"' (~["\r\n])* '"' | '\'' (~['\r\n])* '\'' ;

CONSTANT: 'pi' | 'e' | 'inf' ;

ID: [a-zA-Z_][a-zA-Z0-9_]* ;

// 运算符
POW: '**' ;
LSHIFT: '<<' ;
RSHIFT: '>>' ;

// 忽略空白字符
WS: [ \t\r\n]+ -> skip ;

fragment DIGIT: [0-9] ; 