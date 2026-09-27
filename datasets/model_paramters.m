% This file enlists all the model paramters of the QTank system

QTank.a1 = 0.071 ;    % cm^2 (Area of outlet tube for Tank 1)
QTank.A1 = 28    ;    % cm^2 (Area of tank 1)

QTank.a3 = 0.071 ;    % cm^2 (Area of outlet tube for Tank 3)
QTank.A3 = 28    ;    % cm^2 (Area of tank 3)

QTank.a2 = 0.057 ;    % cm^2 (Area of outlet tube for Tank 2)
QTank.A2 = 32    ;    % cm^2 (Area of tank 2)

QTank.a4 = 0.057 ;    % cm^2 (Area of outlet tube for Tank 4)
QTank.A4 = 32    ;    % cm^2 (Area of tank 4)

QTank.gam1 = 0.43 ;    % cm^3/Vs
QTank.gam2 = 0.34 ;    % cm^3/Vs
QTank.gam3 = 0.4  ;    % For disturbance input 

QTank.k1 =   3.14 ;
QTank.k2 =   3.29 ;

% Standard deviations of measurement noise 
QTank.Q_mat =  diag([ 0.05^2 ])' ;   
% Standard deviations of Unmeasured white noise disturbance in inputs 
QTank.R_mat =  diag([ 0.05^2 0.05^2 ]') ;  