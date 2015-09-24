        %% Stacked Random Forests:
         % Concept: Train an ensemble of random forests on individual time
         % bins across all neurons, then train a final random forest based
         % on the output of the ensemble.
        
        % Each forest generates a label (between 1-4) which is converted
        % into a binary code (e.g. 1000 for label 1, 0100 for label 2, etc). 
        % These probabilities are then used to grow a final random forest. 
        
        load testData
        
        nTrees = 500;
        classIDs = unique(y);
        nTest    = length(yTest);
        
        %% check that data is consistent:
        mod   = classRF_train(squeeze(xTrain(:,5,:)),yTrain,nTrees);
        yHat  = classRF_predict(squeeze(xTest(:,iBin,:)),mod);
        perc = sum( yTest == yHat )/nTest
        %%(if all correct, this should give a result around perc = 0.41)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% TRAINING
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [nEx,nBin,nFeat] = size(xTrain);
        unitEstimate = zeros(nEx,nBin);
        % train random forest (using the mex version) by each time bin:
        for iBin = 1:nBin
            model{iBin} = classRF_train(squeeze(xTrain(:,iBin,:)),yTrain,nTrees);
            unitEstimate(:,iBin)  = classRF_predict(squeeze(xTrain(:,iBin,:)),model{iBin});
        end
        %%(note that after training and testing on the same data, the labels are all the same and all correct across time bins within unitEstimate)
        
        % Each unit at each trial contributes to a label, which should be rewritten
        % as a binary code (e.g. 1000 for label 1, 0100 for label 2, etc). These codes are rolled out across units for
        % each trial:
        allUnitEstDigi = [];
        for iBin = 1:nBin
            unitEstDigi = zeros(nEx,length(classIDs));
            yHats = unitEstimate(:,iBin);
            unitEstDigi(yHats==1,1) = 1;
            unitEstDigi(yHats==2,2) = 1;
            unitEstDigi(yHats==3,3) = 1;
            unitEstDigi(yHats==4,4) = 1;
            
            allUnitEstDigi = [allUnitEstDigi, unitEstDigi];
        end
            
        %train ultimate random forest using the reduced data:
        model2 = classRF_train(allUnitEstDigi,yTrain,nTrees);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% TESTING
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [nEx,nBin,nFeat] = size(xTest);
        unitEstimate2 = zeros(nEx,nFeat);
        % train random forest (using the mex code)
        for iBin = 1:nBin
            unitEstimate2(:,iBin)  = classRF_predict(squeeze(xTest(:,iBin,:)),model{iBin});
        end
        
        allUnitEstDigi = [];
        for iBin = 1:nBin
            unitEstDigi = zeros(nEx,length(classIDs));
            yHats = unitEstimate2(:,iBin);
            unitEstDigi(yHats==1,1) = 1;
            unitEstDigi(yHats==2,2) = 1;
            unitEstDigi(yHats==3,3) = 1;
            unitEstDigi(yHats==4,4) = 1;
            
            allUnitEstDigi = [allUnitEstDigi, unitEstDigi];
        end

        yHat = classRF_predict(allUnitEstDigi,model2);
        perc = sum( yTest == yHat )/nTest
        %digital labels from first layer of random forests: about the same
        %performance as a good time bin alone, with a single forest.
