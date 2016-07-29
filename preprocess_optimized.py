import numpy
import sys
import pickle
f = open("kddcup.names","r")
counter = 1
labels = {}; features = {}
for line in f:
	line = line.strip() #get rid of period at end
	if counter == 1:
		s = line.split(",") #split first line on comma
		for label in s:
			labels[label] = 0  #each CSV in first line is label, add to 
					   #label dictionary as key with value 0
	else:
		s = line.split(":")   #split second line on ':' add name of feature
					#as key to features dictionary
		features[s[0]]=0
	counter += 1
f.close()

#print labels.keys(), len(labels.keys())
#print features.keys(), len(features.keys())


f = open("kddcup.data.corrected","r")
data_dict = {}
durations = 0
src_bytes = 0
dest_bytes = 0
wrong_fragments = 0
urgent = 0
hot = 0
fail = 0
compromised = 0
root = 0
file_create = 0
shells = 0
access = 0
outbound_cmds = 0
count = 0
srv_count = 0
#serror = []
dest_hosts = 0
dest_host_srv = 0

for label in labels:
	data_dict[label] = []
	  #make each value of label a list

print "Reading the data file..."

line_counter = 0

for line in f:
    line_counter += 1
    if line_counter % 200000 == 0:
        print str(line_counter)+" lines processed."
    line = line.strip();  
    line = line[:-1]     #get rid of period at end
    s = line.split(",")  #split line on commas
    try:
        labels[s[-1]] += 1  #increment label everytime it occurs in file
        data_dict[s[-1]].append(s[:-1]) #append line to value of correct label
        #durations.append(float(s[0]))
        if (float(s[0]) > durations):
            durations = float(s[0])
        #src_bytes.append(float(s[4]))
        if (float(s[4]) > src_bytes):
            src_bytes = float(s[4])
        #dest_bytes.append(float(s[5]))
        if (float(s[5]) > dest_bytes):
            dest_bytes = float(s[5])
    
        #wrong_fragments.append(float(s[7]))
        if (float(s[7]) > wrong_fragments):
            wrong_fragments = float(s[7])
        #urgent.append(float(s[8]))
        if (float(s[8]) > urgent):
            urgent = float(s[8])
        #hot.append(float(s[9]))
        if (float(s[9]) > hot):
            hot = float(s[9])
        #fail.append(float(s[10]))
        if (float(s[10]) > fail):
            fail = float(s[10])
        #compromised.append(float(s[12]))
        if (float(s[12]) > compromised):
            compromised = float(s[12])
        #root.append(float(s[15]))
        if (float(s[15]) > root):
            root = float(s[15])
        #file_create.append(float(s[16]))
        if (float(s[16]) > file_create):
            file_create = float(s[16])
        #shells.append(float(s[17]))
        if (float(s[17]) > shells):
            shells = float(s[17])
        #access.append(float(s[18]))
        if (float(s[18]) > access):
            access = float(s[18])
        #outbound_cmds.append(float(s[19]))
        if (float(s[19]) > outbound_cmds):
            outbound_cmds = float(s[19])
        #count.append(float(s[22]))
        if (float(s[22]) > count):
            count = float(s[22])
        #srv_count.append(float(s[23]))
        if (float(s[23]) > srv_count):
            srv_count = float(s[23])
        #dest_hosts.append(float(s[31]))
        if (float(s[31]) > dest_hosts):
            dest_hosts = float(s[31])
        #dest_host_srv.append(float(s[32]))
        if (float(s[32]) > dest_host_srv):
            dest_host_srv = float(s[32])
        #for i in range(len(s)-1):
            #features[s[i]]+=1
    except KeyError:
        pass
    
f.close()

print str(line_counter)+ " lines processed, start quantifying..."
 
dur_max = durations
src_bytes_max = src_bytes
dest_bytes_max = dest_bytes
wrong_fragments_max = wrong_fragments
urgent_max = urgent
hot_max = hot
fail_max = fail
compromised_max = compromised
root_max = root
file_create_max = file_create
shell_max = shells
access_max = access
outbound_max = outbound_cmds
count_max = count
srv_count_max = srv_count
dest_host_max = dest_hosts
dest_host_srv_max = dest_host_srv

min_items = 10000             #Minimum number of examples per class (very important variable)
random_dict = {}
for label in labels:
	print("here", label, labels[label])
	if(labels[label] >= min_items): # for each label, if it has more than min_items
				# create a key for random_dict and list as value
		random_dict[label] = []


rng = numpy.random.RandomState(11)

for label in random_dict:
	indice = rng.choice(range(0,len(data_dict[label])),min_items,replace = False)
	# generate random list of indices of size min_items
	for elem in indice:
		random_dict[label].append(data_dict[label][elem])
	#from each label in random_dict, append to random_dict[label] the random indices

#for label in random_dict:
#	for line in random_dict[label]:		# remove unwanted features
#		print line[0]
#		line.pop(1)

for label in random_dict:
	for line in random_dict[label]:
		if line[1] == "tcp":
			line[1]= 1/6.0
		elif line[1] == "udp":
			line[1]= 2/6.0		# assign values to second column of value lists (protocol type)
		else:
			line[1]= .5

services = ['http', 'smtp', 'finger', 'domain_u', 'auth',\
             'telnet', 'ftp', 'eco_i', 'ntp_u', 'ecr_i', \
             'other', 'private', 'pop_3', 'ftp_data', \
             'rje', 'time', 'mtp', 'link', 'remote_job', \
             'gopher', 'ssh', 'name', 'whois', 'domain', \
             'login', 'imap4', 'daytime', 'ctf', 'nntp', \
             'shell', 'IRC', 'nnsp', 'http_443', 'exec', \
             'printer', 'efs', 'courier', 'uucp', 'klogin', \
             'kshell', 'echo', 'discard', 'systat', 'supdup', \
             'iso_tsap', 'hostnames', 'csnet_ns', 'pop_2', \
             'sunrpc', 'uucp_path', 'netbios_ns', 'netbios_ssn', \
             'netbios_dgm', 'sql_net', 'vmnet', 'bgp', 'Z39_50', \
             'ldap', 'netstat', 'urh_i', 'X11', 'urp_i', 'pm_dump', \
             'tftp_u', 'tim_i', 'red_i','SF', 'aol','http_8001', 'harvest']
             
flags = ['S1', 'SF', 'REJ', 'S2', 'S0', 'S3', 'RSTO', 'RSTR', 'RSTOS0', 'OTH', 'SH']
for label in random_dict:
	for List in random_dict[label]:
			for i in range(len(List)):
				if i == 0:
					List[i] = float(List[i]) / float(dur_max) # assign values to first column of value lists (duration)
				elif i == 2:
					List[i]= float(services.index(List[i]))/float(len(services))   # assign values to third column of value lists (service)
				elif i == 3:
					List[i] = float(flags.index(List[i]))/float(len(flags))# assign values to fourth column of value lists (flags)
				elif i == 4:
					List[i] = float(List[i]) / float(src_bytes_max)	# assign values to fifth column of value lists (src_bytes)
				elif i == 5:
					
					List[i] = float(List[i]) / float(dest_bytes_max)	# assign values to sixth column of value lists (dest_bytes)
				elif i == 6:
					List[i] = float(List[i])	# convert seventh  column of value lists to float (land)
				elif i == 7:
					List[i] = float(List[i]) / float(wrong_fragments_max)	# assign values to eighth column of value lists (wrong_fragments)
				elif i == 8:
					List[i] = float(List[i]) / float(urgent_max)	# assign values to ninth column of value lists (urgent)
				elif i == 9:
					List[i] = float(List[i]) / float(hot_max)	# assign values to 10th column of value lists (hot)
				elif i == 10:
					List[i] = float(List[i]) / float(fail_max)	# assign values to 11th column of value lists (failed_logins)
				elif i == 11:
					List[i] = float(List[i])	# convert 12th column (successful login) to float
				elif i == 12:
					List[i] = float(List[i]) / float(compromised_max)	# assign values to 13th column of value lists (num_comprimesed)
				elif i == 13:
					List[i] = float(List[i])	# convert 14th column (root shell) to float
				elif i == 14:
					List[i] = float(List[i])	# convert 15th column (su_attemptedl) to float
				elif i == 15:
					List[i] = float(List[i]) / float(root_max)	# assign values to 16th column of value lists (num_root)
				elif i == 16:
					List[i] = float(List[i]) / float(file_create_max)	# assign values to 17th column of value lists (num_file creations)
				elif i == 17:
					List[i] = float(List[i]) / float(shell_max)	# assign values to 18th column of value lists (num_shells)
				elif i == 18:
					List[i] = float(List[i]) / float(access_max)	# assign values to 19th column of value lists (num_access_files)
				elif i == 19:
					   #get rid of becasue all 0
					List[i] =  float(List[i])/ float(sys.maxint)	# assign values to 20th column of value lists (num_outbound_commands)
				elif i == 20:
					List[i] = float(List[i])	# convert 21st column (is_host_login) to float

				elif i == 21:
					List[i] = float(List[i])	# convert 22nd column (is_guest_login) to float
				elif i == 22:
					List[i] = float(List[i]) / float(count_max)	# assign values to 23rd column of value lists (count)

				elif i == 23:
					List[i] = float(List[i]) / float(srv_count_max)	# assign values to 24th column of value lists (srv_count)
				elif i == 24:
					List[i] = float(List[i]) 	# convert 25th column (serror_rate) to float 
				elif i == 25:
					List[i] = float(List[i]) 	# convert 26th column (srv_serror_rate) to float 		
				elif i == 26:
					List[i] = float(List[i]) 	# convert 27th column (rerror_rate) to float
				elif i == 27:
					List[i] = float(List[i]) 	# convert 28th column (srv_rerror_rate) to float
				elif i == 28:
					List[i] = float(List[i]) 	# convert 29th column (same_srv_rate) to float
				elif i == 29:
					List[i] = float(List[i]) 	# convert 30th column (diff_srv_rate) to float
				elif i == 30:
					List[i] = float(List[i]) 	# convert 31st column (srv_diff_host_rate) to float
				elif i == 31:
					List[i] = float(List[i]) / float(dest_host_max) 	# assign values to 32nd column of value lists (dest_host_count)
				elif i == 32:
					List[i] = float(List[i]) / float(dest_host_srv_max) 	# assign values to 33rd column of value lists (dst_host_srv_count)
				elif i == 33:
					List[i] = float(List[i]) 	# convert 34th column (dst_host_same_srv_rate) to float
				elif i == 34:
					List[i] = float(List[i]) 	# convert 35th column (dst_host_diff_srv_rate) to float
				elif i == 35:
					List[i] = float(List[i]) 	# convert 36th column (dst_host_same_src_port_rate) to float
				elif i == 36:
					List[i] = float(List[i]) 	# convert 37th column (dst_host_srv_diff_host_rate) to float
				elif i == 37:
					List[i] = float(List[i]) 	# convert 38th column (dst_host_serror_rate) to float
				elif i == 38:
					List[i] = float(List[i]) 	# convert 39th column (dst_host_srv_serror_rate) to float
				elif i == 39:
					List[i] = float(List[i]) 	# convert 40th column (dst_host_rerror_rate) to float
				elif i == 40:
					List[i] = float(List[i]) 	# convert 41st column (dst_host_srv_rerror_rate) to float



#print random_dict

file2 = open("labeling.txt","w")
print "Generating training set..."

all_data = []; labels = random_dict.keys()[:]
for k in labels:
	file2.write(str(k))
	for data in random_dict[k]:
		all_data.append((data,labels.index(k)))

print "Pickling..."
with open('all_data_' + str(min_items) +'.pickle','w') as f:
	pickle.dump(all_data,f)
file2.close
