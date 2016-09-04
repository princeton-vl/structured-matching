-- test code for structured matching
require 'nn'
require 'hdf5'
import 'solver_pc_test.lua'

cmd = torch.CmdLine()
cmd:option('-lamda' , 0.1)
cmd:option('-file' , '../../workspace/model/matching_pc.asc')
params = cmd:parse(arg)
print(params)
method = 'matching_pc'
outpath = '../../workspace'
outpath = string.format("%s/%s/" , outpath , method)
if not path.exists(outpath) then
        os.execute("mkdir " .. outpath)
end

splist = {9}
num_cls = #splist
function load_image_feat()
        local file = hdf5.open('../../workspace/test/test_feat_vgg_without_ft.mat' , 'r')
        local data = file:read('feat_vgg'):all()
        print (#data)
        file:close()
        return data:transpose(3,1):double()
end
function load_box()
        local file = hdf5.open('../../workspace/test/edgebox_test.mat' , 'r')
        local data = file:read('box'):all()
        print (#data)
        file:close()
        return data:transpose(2,1):double()+1
end
function load_idx()
        local file = hdf5.open('../../workspace/test/test_idx.mat' , 'r')
        local data = file:read('test_sen_idx'):all()
        print (#data)
        file:close()
        return data:transpose(2,1)
end
function load_phrase_feat()
        local file = hdf5.open('../../workspace/test/testpv_pca.mat' , 'r')
        local data = file:read('testpv_pca'):all()
        print (#data)
        file:close()
        return data:transpose(2,1)
end
function write_to_file(fp , out)
	local fw = io.open(fp , 'w')
	for i=1,out:size()[1] do
		fw:write(string.format("%f" , out[{i,1}]))
		for j=2,out:size()[2] do
			fw:write(string.format(" %d" , out[{i,j}]))
		end
		fw:write("\n")
	end
	fw:close()
end
function load_sp()
        local file = hdf5.open('../../workspace/test/testsp.mat','r')
        local posgeo = file:read('spposgeo'):all():transpose(2,1)
        local neggeo = file:read('spneggeo'):all():transpose(2,1)
        local posidx = file:read('spposidx'):all():transpose(2,1)
        local negidx = file:read('spnegidx'):all():transpose(2,1)
        local poscnt = file:read('poscnt'):all():transpose(2,1)
        local negcnt = file:read('negcnt'):all():transpose(2,1)
        file:close()
        file = hdf5.open('../../workspace/test/train_sen2pvidx.mat' , 'r')
        local sen2pvidx = file:read('sen2pvidx'):all():transpose(3,1)
        file:close()
        return posgeo,neggeo,posidx,negidx,poscnt,negcnt,sen2pvidx
end
function relation4sen()
        local sp4sen_cnt = torch.zeros(1000,5)
        local sp4sen = {}
        local img,sen,p1,p2
        for i = 1,1000 do
                sp4sen[i] = {}
                for j = 1,5 do
                        sp4sen[i][j] = {}
                        sp4sen_cnt[{i,j}] = 0
                end
        end
        for k = 1,num_cls do
                local i = splist[k]
                for j = poscnt[{i,1}]+1,poscnt[{i+1,1}] do
                        img = posidx[{j,1}]
                        sen = posidx[{j,2}]
                        p1 = posidx[{j,3}]
                        p2 = posidx[{j,4}]
                        sp4sen_cnt[{img,sen}] = sp4sen_cnt[{img,sen}]+1
                        sp4sen[img][sen][sp4sen_cnt[{img,sen}]] = {i,p1,p2}
                end
        end
        return sp4sen,sp4sen_cnt
end
function cal_score(pred,ix,bidx,sen)
        local score = {}
        local ip,sp,l1,l2,r1,r2,r,p1,p2,b1,b2,im,num_r,_b1,_b2,t1,t2
        local num_b = 1
        local cnt1,cnt2,cnt3 = 0,0,0
        local rtype = torch.Tensor(50000)
	local feat = torch.Tensor(50000,dim_img*2)
        for i = 1,num_b do
		im = bidx
                ip = torch.mm(pred[1],pred[2]:transpose(2,1))
                _,sortc = torch.sort(ip,1,true)
                sp,spgt = torch.Tensor(2000,6),torch.Tensor(20,6)
                num_r,num_rgt = 0,0
                for j = 1,sp4sen_cnt[{im,sen}] do
			t2 = num_r
			t1 = cnt3
                        r,p1,p2 = sp4sen[im][sen][j][1],sp4sen[im][sen][j][2],sp4sen[im][sen][j][3]
                        for k1 = 1,10 do
                 		for k2 = 1,10 do
					b1,b2=p1,p2
            	                        _b1 = sortc[{k1,b1}]
             				_b2 = sortc[{k2,b2}]
               		    	        cnt3 = cnt3+1
              	        	      	rtype[cnt3] = r
                      	    	        feat[{cnt3,{1,dim_img}}] = ix[cnt1+_b1]
      		      			feat[{cnt3,{dim_img+1,dim_img*2}}] = ix[cnt1+_b2]
                            	        num_r = num_r+1
                      	                sp[num_r] = torch.Tensor({_b1,_b2,b1,b2,0,-1})
            	                end
     	                end
                        sp[{{t2+1,num_r},5}] = logis_cnn[r]:forward(feat[{{t1+1,cnt3},{}}])
                end
                if num_r == 0 then sp = torch.Tensor({-1}) else sp = sp[{{1,num_r},{}}] end
                score[i] = {torch.Tensor(ip:size()):copy(ip) , torch.Tensor(sp:size()):copy(sp)}
        end
        return score,sortc
end

--load data
feat_pv = load_phrase_feat()
feat_vgg = load_image_feat()
idx = load_idx()
box = load_box()
posgeo,neggeo,posidx,negidx,poscnt,negcnt,sen2pvidx = load_sp()
sp4sen,sp4sen_cnt = relation4sen()
file = torch.DiskFile(params.file , 'r')
prl_all = file:readObject()
prl = prl_all[1]
logis_cnn = prl_all[2]
prl:evaluate()
--global parameters
dim_img = 4096
dim_pv = 6000
num_sample = feat_vgg:size()[1]
num_sen = 5
fid = io.open('../../workspace/test/testID.txt' , 'r')
out = torch.Tensor(feat_vgg:size()[2],5)
cnt = 0
for i = 1,num_sample do
	id = fid:read()
	for j=1,num_sen do
		print(i,j)
		vgg = feat_vgg[i]
		pv = feat_pv[{{cnt+1,cnt+idx[{i,j}]},{}}]
		cnt = cnt+idx[{i,j}]
		pred = prl:forward({vgg,pv})
		score,sortc = cal_score(pred,vgg,i,j)
		_loss,_y = solve_matching(score[1] , torch.zeros(pv:size()[1]) , params.lamda)	
		nr,np = vgg:size()[1],pv:size()[1]
		for k=1,pv:size()[1] do
			fp = string.format("%s%s_%d_%d.txt" , outpath , id , j , k)
			out[{{},{1}}] = torch.zeros(vgg:size()[1] , 1)
			_max,_maxidx,_n=-1,-1,0
			for l=1,vgg:size()[1] do
				if _y[(l-1)*np+k] > _max then
					_max = _y[(l-1)*np+k]
					_maxidx = l
					_n = _n+1
				end
			end
			out[{_maxidx,1}] = _loss
			out[{{},{2,5}}] = box[{{(i-1)*100+1,i*100},{}}]
			write_to_file(fp , out)	
		end
	end
	if i % 10 == 0 then
		print(i)
	end	
end


