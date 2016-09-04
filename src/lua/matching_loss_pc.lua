-- loss function for structured matching
require 'nn'
require 'solver_pc_train.lua'
local matching_loss, parent = torch.class('nn.matching_loss', 'nn.Module')
function matching_loss:__init(lamda)
        self.lamda = lamda
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
	self.score = input[5]
	local batch = idxx:size()[1]
	local len = 0
	local cnt1,cnt2,l1,l2,r1,r2
	cnt1=0
	cnt2=0
	self.tpv,self.allv = 0,0
	input1, input2 = makeContiguous(self, input1, input2)
	self.pred = {}
	self.output = 0
	--calculate matching and loss for each pair	
	for i = 1,batch do
		l1 = cnt1+1
		r1 = idxx[i]
		l2 = cnt2+1
		r2 = idxy[i]	
		-- remember to change the index of gt to 1,2,3...
		local _loss,_y = solve_matching(self.score[i] , gtb[{{l2,r2}}] , self.lamda)
		self.output = self.output+_loss
		self.pred[i] = _y -- here we store index 1,2,3
		cnt2 = r2
		cnt1 = r1
	end
	return self.output
end

function matching_loss:updateGradInput(input)
	local input1, input2 = input[1][1], input[1][2]
        local idxx, idxy, gtb = input[2], input[3] , input[4]
        local batch = idxx:size()[1]
        local len = 0
        local cnt1,cnt2,cnt3,cnt4,l1,l2,r1,r2,t1,t2,np,nr,W,sp,nsp
        cnt1=0
        cnt2=0
	cnt4=0
	local gw1 = torch.Tensor(input1:size()):fill(0)
	local gw2 = torch.Tensor(input2:size()):fill(0)
	local gsp = torch.Tensor(50000):fill(0)
	self.gradInput = {}	
	for i = 1,batch do
		l1 = cnt1+1
                r1 = idxx[i]
		l2 = cnt2+1
               	r2 = idxy[i]
		W,sp = self.score[i][1],self.score[i][2]
		nr,np = W:size()[1],W:size()[2]
		cnt3 = 0
		if sp:size():size() == 1 then nsp = 0 else nsp = sp:size()[1] end
		for j1 = 1,nr do 
			for j2 = 1,np do
				cnt3 = (j1-1)*np+j2	
				if self.pred[i][cnt3] > 1e-8 then
					t2 = j2+l2-1	
					t1 = j1+l1-1
					gw1[t1] = gw1[t1]+input2[t2]*self.pred[i][cnt3]
					gw2[t2] = gw2[t2]+input1[t1]*self.pred[i][cnt3]
					if self.pred[i][cnt3] > 0.9 and gtb[t2] == j1 then self.tpv = self.tpv+1 end
				end
			end
		end
		for j = l2,r2 do
			if gtb[j] > 0 then 
				t2 = gtb[j]+l1-1
				gw1[t2] = gw1[t2]-input2[j]
				gw2[j] = gw2[j]-input1[t2]
			end	
			self.allv = self.allv + 1
		end	
		if nsp > 0 then
			for j = 1,nsp do 
				t1 = np*nr+np+nr+j*4-3
				if self.pred[i][t1] > 1e-8 then
					gsp[cnt4+j] = gsp[cnt4+j]+self.pred[i][t1]*self.lamda
				end
				if sp[{j,6}] == 1 then
					gsp[cnt4+j] = gsp[cnt4+j]-self.lamda
				end
			end
			cnt4 = cnt4+nsp
		end
		cnt2 = r2
		cnt1 = r1
	end
	gsp = gsp[{{1,cnt4}}]
	self.gradInput[1] = {}	
	self.gradInput[2] = torch.Tensor(gsp:size()):copy(gsp)
	self.gradInput[1][1] = torch.Tensor(gw1:size()):copy(gw1)
        self.gradInput[1][2] = torch.Tensor(gw2:size()):copy(gw2)

        return self.gradInput
end
