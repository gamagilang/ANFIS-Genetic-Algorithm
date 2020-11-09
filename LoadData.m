%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML122
% Project Title: Feature Selection using GA (Variable Number of Features)
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function data=LoadData()

    dataset=load('bodyfat_data');
    data.x=dataset.Inputs;
    data.t=dataset.Targets;
    
    data.nx=size(data.x,1);
    data.nt=size(data.t,1);
    data.nSample=size(data.x,2);

end