-- loss function for bipartite matching
require 'nn'
require 'solver.lua'
local matching_loss, parent = torch.class('nn.matching_loss', 'nn.Module')
function matching_loss:__init(m)
        self.m = m
        parent.__init(self)
        self.gradInput = {torch.Tensor(), torch.Tensor()}
end


local function makeContiguous(self, input1, input2)
   if not input1:isContiguous() then
      self._input1 = self._input1 or input1.new()
      self._input1:resizeAs(input1):copy(input1)
      input1 = self._input1
   end
   if not input2:isContiguous() then
      self._input2 = self._input2 or input2.new()
      self._input2:resizeAs(input2):copy(input2)
      input2 = self._input2
   end
   return input1, input2
end

function matching_loss:updateOutput(input)
	local input1, input2 = input[1][1], input[1][2]
	local idxx, idxy, gtb = input[2], input[3] , input[4]
	local batch = idxx:size()[1]
	local len = 0
	local cnt1,cnt2,l1,l2,r1,r2
	cnt1=0
	cnt2=0
	self.tpv,self.tpnv,self.allv,self.allnv = 0,0,0,0
	input1, input2 = makeContiguous(self, input1, input2)
	self.pred = torch.Tensor(gtb:size())
	self.output = 0
	--calculate matching and loss for each pair	
	for i = 1,batch do
		l1 = cnt1+1
		r1 = idxx[i]
		if idxy[i] > cnt2 then
			l2 = cnt2+1
			r2 = idxy[i]	
			self.W = torch.mm(input1[{{l1,r1},{}}] , input2[{{l2,r2},{}}]:transpose(2,1))
			-- remember to change the index of gt to 1,2,3...
			local _loss,_y = solve_matching(self.W , gtb[{{l2,r2}}] , self.m)
			self.output = self.output+_loss
			self.pred[{{l2,r2}}] = _y -- here we store index 1,2,3
			cnt2 = r2
		end
		cnt1 = r1
	end
	return self.output
end

function matching_loss:updateGradInput(input)
	local input1, input2 = input[1][1], input[1][2]
        local idxx, idxy, gtb = input[2], input[3] , input[4]
        local batch = idxx:size()[1]
        local len = 0
        local cnt1,cnt2,l1,l2,r1,r2,t1,t2
        cnt1=0
        cnt2=0
	local gw1 = torch.Tensor(input1:size()):fill(0)
	local gw2 = torch.Tensor(input2:size()):fill(0)
	
	for i = 1,batch do
		l1 = cnt1+1
                r1 = idxx[i]
		if idxy[i] > cnt2 then
			l2 = cnt2+1
                	r2 = idxy[i]
			for j = l2,r2 do
				if gtb[j] ~= self.pred[j] then
					if self.pred[j] > 0 then
						t1 = self.pred[j]+l1-1
						gw1[t1] = gw1[t1]+input2[j]
						gw2[j] = gw2[j]+input1[t1]
					end
					if gtb[j] > 0 then
						t2 = gtb[j]+l1-1
						gw1[t2] = gw1[t2]-input2[j]
						gw2[j] = gw2[j]-input1[t2]
					end
				else
					if gtb[j] == 0 then
						self.tpnv = self.tpnv+1
					else
						self.tpv = self.tpv+1
					end
				end
				if gtb[j] == 0 then
					self.allnv = self.allnv+1
				else
					self.allv = self.allv+1
				end
			end
			cnt2 = r2
		end
		cnt1 = r1
	end
	
	self.gradInput[1]:resize(gw1:size()):copy(gw1)
        self.gradInput[2]:resize(gw2:size()):copy(gw2)

        return self.gradInput
end
