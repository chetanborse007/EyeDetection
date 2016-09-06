

function [left_x, right_x, left_y, right_y] = eye_detection(img)
% DESCRIPTION: Algorithm to detect Left and Right Eye.
% INPUT:       RGB image
% OUTPUT:      X and Y coordinates of Left and Right Eye.

    % Create a Pattern Recognition Network and train it.
    % [Optional] Test trained Pattern Recognition Network using few test
    % images specified.
    if exist(fullfile(cd, 'PatternNet.mat'), 'file') ~= 2
        [TrainX, TrainY] = PrepareTrainingSet();
        CreatePatternNet([20 40 80 40 20], true);
        TrainPatternNet(TrainX, TrainY);
        
        isTest = false;
        if isTest
            TestPatternNet();
        end
    end

    % Find width and height of input image
    Width  = size(img, 2);
    Height = size(img, 1);

    % Transform image from RGB colorspace to LAB colorspace
    RGB2LAB = makecform('srgb2lab');
    lab_img = applycform(img, RGB2LAB);
    
    % Use 'L' component of LAB image, where 'L' stands for lightness
    % contrast.
    % As Eyedot corresponds to brightest spot in face image, 
    % hence using 'L' component returns better results.
    L = lab_img(:, :, 1);
    
    % Detect strongest corner points in image using FAST algorithm.
    % Expected Eye Candidate can be one among retrieved corners.
    Corners       = detectFASTFeatures(L);
    EyeCandidates = Corners.selectStrongest(50);
    %figure;imshow(img);
    %hold on;
    %plot(EyeCandidates);

    % Follow below steps for every strongest Eye Candidate (as specified by
    % 'EyeCandidates')
    LeftEye      = [];
    RightEye     = [];
    EyeDot       = [];
    LeftEyeProb  = [];
    RightEyeProb = [];
    for i = 1:EyeCandidates.length
        % Get X and Y coordinates of Eye Candidate under inspection
        point = EyeCandidates.Location(i, :);
        X = point(:, 1);
        Y = point(:, 2);
        
        % Crop proportional region surrounding Eye Candidate location
        X1 = X - (Width/10);
        Y1 = Y - (Height/20);
        EyeCandidate = imcrop(img, [X1, Y1, Width/5, Height/10]);
        %figure;imshow(EyeCandidate);

        % Preprocess Eye Candidate image
        test_img = PreprocessImage(EyeCandidate);
        
        % Predict whether it is Left Eye, Right Eye or Not an Eye 
        % using a Pattern Recognition Network
        isEye = Predict(test_img);
        
        % Find which output class among 3 has maximum probability 
        % and accordingly record corresponding output class
        [value, index] = max(isEye);
        if index == 2
            LeftEye  = [ LeftEye i ];
        elseif index == 3
            RightEye = [ RightEye i ];
        end

        % Store Left Eye and Right Eye probabilities for Eye Candidate
        % under inspection
        EyeDot       = [ EyeDot point(:) ];
        LeftEyeProb  = [ LeftEyeProb isEye(2) ];
        RightEyeProb = [ RightEyeProb isEye(3) ];
    end

    % Left Eye:
    % 1. Check which Eye Candidate predicted as a Left Eye has highest
    % Left Eye probability and mark it as final expected Left Eye 
    % Candidate.
    % 2. If 'LeftEye' is empty, then pick an Eye Candidate having highest
    % Left Eye probability and mark it as final expected Left Eye
    % Candidate.
    % 3. If 'LeftEyeProb' is empty, then set default location for Left Eye
    % Candidate.
    if ~isempty(LeftEye)
        [value, index] = max(LeftEyeProb(LeftEye));
        LeftEyeDot     = EyeDot(:, LeftEye(index));
    elseif ~isempty(LeftEyeProb)
        [value, index] = max(LeftEyeProb);
        LeftEyeDot     = EyeDot(:, index);
    else
        disp('Unable to find corner points for Left Eye!!');
        disp('Setting Left Eye Location to default value!!');
        LeftEyeDot     = [ 3.5 * (Width / 10); ...
                           12 * (Height / 32) ];
    end
    
    % Right Eye:
    % 1. Check which Eye Candidate predicted as a Right Eye has highest
    % Right Eye probability and mark it as final expected Right Eye 
    % Candidate.
    % 2. If 'RightEye' is empty, then pick an Eye Candidate having highest
    % Right Eye probability and mark it as final expected Right Eye
    % Candidate.
    % 3. If 'RightEyeProb' is empty, then set default location for a Right 
    % Eye Candidate.
    if ~isempty(RightEye)
        [value, index] = max(RightEyeProb(RightEye));
        RightEyeDot    = EyeDot(:, RightEye(index));
    elseif ~isempty(RightEyeProb)
        [value, index] = max(RightEyeProb);
        RightEyeDot    = EyeDot(:, index);
    else
        disp('Unable to find corner points for Right Eye!!');
        disp('Setting Right Eye Location to default value!!');
        RightEyeDot    = [ 6.5 * (Width / 10); ...
                           12 * (Height / 32) ];
    end

    % Set X and Y coordinates for Left Eye and Right Eye
    left_x  = LeftEyeDot(1, :);
    left_y  = LeftEyeDot(2, :);
    right_x = RightEyeDot(1, :);
    right_y = RightEyeDot(2, :);

    % Display X and Y coordinates for Left Eye and Right Eye
    isShow = false;
    if isShow
        disp(['Left Eye: ', LeftEyeDot]);
        disp(['Right Eye: ', RightEyeDot]);
    end

end


function [TrainX, TrainY] = PrepareTrainingSet()
% DESCRIPTION: Prepare training data for Pattern Recognition Network.
% INPUT:       None
% OUTPUT:      Training images (TrainX) and 
%              its corresponding labels (TrainY).

    % Input folders for training data, 
    % which are organized in accordance with labels.
    NonEyeSet   = './Train/NonEye/';
    LeftEyeSet  = './Train/Eye/Left/';
    RightEyeSet = './Train/Eye/Right/';

    % Read all JPEG images
    NonEyes   = dir([NonEyeSet, '*.jpg']);
    LeftEyes  = dir([LeftEyeSet, '*.jpg']);
    RightEyes = dir([RightEyeSet, '*.jpg']);

    % Image count for every category
    NonEyeCount   = size(NonEyes, 1);
    LeftEyeCount  = size(LeftEyes, 1);
    RightEyeCount = size(RightEyes, 1);
    
    % Preprocessed training images (TrainX) and 
    % its corresponding labels (TrainY)
    TrainX = [];
    TrainY = [];
    
    % For every 'Not an Eye', preprocess image and append it to 'TrainX'.
    % Set label vector [1 0 0].
    for i = 1:NonEyeCount
        rgb_img = imread([ NonEyeSet, NonEyes(i).name ]);
        img     = PreprocessImage(rgb_img);
        class   = [ 1 0 0 ];
        TrainX  = [ TrainX img(:) ];
        TrainY  = [ TrainY class(:) ];
    end
    
    % For every 'Left Eye', preprocess image and append it to 'TrainX'.
    % Set label vector [0 1 0].
    for i = 1:LeftEyeCount
        rgb_img = imread([ LeftEyeSet, LeftEyes(i).name ]);
        img     = PreprocessImage(rgb_img);
        class   = [ 0 1 0 ];
        TrainX  = [ TrainX img(:) ];
        TrainY  = [ TrainY class(:) ];
    end

    % For every 'Right Eye', preprocess image and append it to 'TrainX'.
    % Set label vector [0 0 1].
    for i = 1:RightEyeCount
        rgb_img = imread([ RightEyeSet, RightEyes(i).name ]);
        img     = PreprocessImage(rgb_img);
        class   = [ 0 0 1 ];
        TrainX  = [ TrainX img(:) ];
        TrainY  = [ TrainY class(:) ];
    end

end


function [img] = PreprocessImage(rgb_img)
% DESCRIPTION: Preprocess RGB image for Pattern Recognition Network.
% INPUT:       RGB image
% OUTPUT:      Preprocessed image

    % Transform image from RGB colorspace into HSV colorspace and use 'V'
    % component for further processing.
    % Reason:
    % 1. 'V' component of HSV image provides an chromatic notion of 
    % the intensity of the color, shortly brightness of color.
    % 2. Eyedot in face image corresponds to bright spot 
    % due to light reflecting from it.
    % 3. Hence, using 'V' component for further analysis ensures better
    % results.
    [H, S, V]    = rgb2hsv(rgb_img);

    % Normalize 'V' component of HSV image
    norm_img     = im2double(V);
    
    % Resize 'V' component of HSV image to standard size
    standard_img = imresize(norm_img, [200, 100]);
    
    img          = standard_img;

end


function [] = CreatePatternNet(HiddenLayer, ShowTrainWindow)
% DESCRIPTION: Create a Pattern Recognition Network.
% INPUT:       1. Vector of Hidden Layers & its size
%              2. Flag for displaying training window
% OUTPUT:      None

    % Create an object of Pattern Recognition Network.
    PatternNet = patternnet(HiddenLayer);

    % Split input data into Training, Validation and Testing sets
    PatternNet.divideFcn  = 'dividerand';
    PatternNet.divideMode = 'sample';
    PatternNet.divideParam.trainRatio = 75/100;
    PatternNet.divideParam.valRatio   = 20/100;
    PatternNet.divideParam.testRatio  = 5/100;

    % Set network training function.
    % Here, we are using Scaled Conjugate Gradient backpropagation.
    PatternNet.trainFcn = 'trainscg';
    
    % Set parameter for checking crossentropy performance of neural netowork
    PatternNet.performFcn = 'crossentropy';
    
    % Set tuning parameters for Pattern Recognition Network
    % 1. max_fail: Maximum validation failures
    % 2. epochs: Maximum number of epochs to train
    % 3. lr: Learning rate
    % 4. mc: Momentum constant
    % 5. showWindow: Show training GUI
    PatternNet.trainParam.max_fail   = 8;
    PatternNet.trainParam.epochs     = 100;
    PatternNet.trainParam.lr         = 0.001;
    PatternNet.trainParam.mc         = 0.95;
    PatternNet.trainParam.showWindow = ShowTrainWindow;
    
    % Set regularization parameter for generalization of Pattern 
    % Recognition Network and for avoiding overfitting
    PatternNet.performParam.regularization = 0.5;

    % Save Pattern Recognition Network object
    save('PatternNet.mat', 'PatternNet');

end


function [] = TrainPatternNet(TrainX, TrainY)
% DESCRIPTION: Train a Pattern Recognition Network.
% INPUT:       1. Training images (TrainX)
%              2. Corresponding labels (TrainY)
% OUTPUT:      None

    % Load Pattern Recognition Network object from 'PatternNet.mat'
    load('PatternNet.mat');
    
    % Train Pattern Recognition Network
    PatternNet = train(PatternNet, TrainX, TrainY);

    % Save trained Pattern Recognition Network back to 'PatternNet.mat'
    save('PatternNet.mat', 'PatternNet');

end


function [class] = Predict(input)
% DESCRIPTION: Predict whether the given image is Eye or not.
% INPUT:       Input image
% OUTPUT:      Predicted class
    
    % Load trained Pattern Recognition Network object from 'PatternNet.mat'
    load('PatternNet.mat');
    
    % Predict whether the given image is Eye or not
    class = PatternNet([input(:)]);

end


function [] = TestPatternNet()
% DESCRIPTION: Test trained Pattern Recognition Network 
%              for specified test images.
% INPUT:       None
% OUTPUT:      None

    if isTest
        TestSet   = './Test/';
        Tests     = dir([TestSet, '*.jpg']);
        TestCount = size(Tests, 1);

        for i = 1:TestCount
            Test     = [ TestSet, Tests(i).name ];
            test_img = PreprocessImage(imread(Test));
            isEye    = Predict(test_img);
            disp(Test);
            disp(isEye);
        end
    end

end

