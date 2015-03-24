local MiniBatchSGD = torch.class('nn.MiniBatchSGD')

function MiniBatchSGD:__init(module, criterion)
    self.learningRate = 0.01
    self.learningRateDecay = 0
    self.maxIteration = 25
    self.module = module
    self.criterion = criterion
    self.verbose = true
    self.perBatch = 100
end

function MiniBatchSGD:train(dataset)
    local iteration = 1
    local module = self.module
    local criterion = self.criterion

    print("# MiniBatchSGD: training")

    for epoch = 1, self.maxIteration do
        local currentError = 0
        local shuffleIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
        print("# epoch " .. epoch)

        local batches = math.ceil(dataset:size() / self.perBatch)
        for batch = 1, batches do
            module:zeroGradParameters()
            for i = 1, self.perBatch do
                local getPos = batch * self.perBatch + i
                if getPos > dataset:size() then break end
                local pos = shuffleIndices[getPos]
                local example = dataset[pos]
                local input = example[1]
                local target = example[2]

                module:forward(input)
                local err = criterion:forward(module.output, target)
                local gradOutput = criterion:updateGradInput(module.output, target)
                module:backward(input, gradOutput)
                currentError = currentError + err
            end
            module:updateParameters(self.learningRate)
        end
        print("# training error = " .. currentError / dataset:size())
    end
end
