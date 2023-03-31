#include <iostream>       // std::cout
#include <string>         // std::string
#include <bitset>         // std::bitset
#include <algorithm> 
#include <vector> 
#include <cmath> 
#include <fstream>
#include <sdsl/int_vector.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <chrono>
using namespace std::chrono;

void save_bitvec(sdsl::int_vector<1>& init_bit_vector, std::string& fname){
   sdsl::store_to_file(init_bit_vector, fname);
}

void load_bitvec(sdsl::int_vector<1>& init_bit_vector, std::string& fname){
   sdsl::load_from_file(init_bit_vector, fname);
}

class my_rank_vector{
   sdsl::int_vector<1>* bitvec;
   int chunck_size;
   int subchunck_size;
   int num_chuncks;
   int rank_bits;

   sdsl::int_vector<1> subchunk_bitvecs;
   int subchunck_rank_bits;
   int subchunck_table_offset;
   int chuncks_offset;
   public:
   my_rank_vector(sdsl::int_vector<1>* init_bit_vector){
      bitvec = init_bit_vector;
      int n = bitvec->size();
      chunck_size = ceil(log2(n) * log2(n));
      subchunck_size = ceil(0.5 * log2(n));
      num_chuncks = ceil((float)n / chunck_size);
      rank_bits = ceil(log2(n));
      subchunck_rank_bits = ceil(log2(chunck_size)) + 2;
      chuncks_offset = num_chuncks * rank_bits;

      uint64_t rel_rank = 0;
      int chunck_idx = 0;
      int subchunck_table_entries = ceil(chunck_size / subchunck_size);
      int counter = 0;
      subchunck_table_offset = (subchunck_table_entries + 2) * subchunck_rank_bits;
      subchunk_bitvecs = sdsl::int_vector<1>(chuncks_offset + (subchunck_table_entries + 2) * subchunck_rank_bits * num_chuncks, 0, 1);
      for(int i = 0; i < n; i++){
         if (i % chunck_size == 0){
            subchunk_bitvecs.set_int(chunck_idx * rank_bits, rel_rank, rank_bits);
            chunck_idx += 1;
            sdsl::int_vector<1> subchunck_table;
            counter = 0;
         }

         if ((i % chunck_size) % subchunck_size == 0){
            int rel_subchunck_rank = rel_rank - subchunk_bitvecs.get_int((chunck_idx-1) * rank_bits, rank_bits);

            subchunk_bitvecs.set_int(chuncks_offset + (chunck_idx-1) * subchunck_table_offset + counter * subchunck_rank_bits, rel_subchunck_rank, subchunck_rank_bits);
            counter += 1;
         }
         rel_rank += bitvec->get_int(i, 1);
      }
   }

   my_rank_vector(sdsl::int_vector<1>* init_bit_vector, std::string& fname){
      bitvec = init_bit_vector;
      int n = bitvec->size();
      chunck_size = ceil(log2(n) * log2(n));
      subchunck_size = ceil(0.5 * log2(n));
      num_chuncks = ceil((float)n / chunck_size);
      rank_bits = ceil(log2(n));
      subchunck_rank_bits = ceil(log2(chunck_size)) + 2;
      chuncks_offset = num_chuncks * rank_bits;

      uint64_t rel_rank = 0;
      int chunck_idx = 0;
      int subchunck_table_entries = ceil(chunck_size / subchunck_size);
      int counter = 0;
      subchunck_table_offset = (subchunck_table_entries + 2) * subchunck_rank_bits;
      subchunk_bitvecs = sdsl::int_vector<1>(chuncks_offset + (subchunck_table_entries + 2) * subchunck_rank_bits * num_chuncks, 0, 1);

      sdsl::load_from_file(subchunk_bitvecs, fname);   
   }

   uint64_t rank1(uint64_t i){
      int chunck_idx = floor(i / chunck_size);
      int chunck_rank = subchunk_bitvecs.get_int(chunck_idx * rank_bits, rank_bits); //rel_cum_ranks[chunck_idx];
      int offset = chunck_size * chunck_idx;

      int subchunck_idx = floor((float)(i - offset) / subchunck_size);
      int subchunck_offset = subchunck_idx * subchunck_size;
      int subchunck_rank = subchunk_bitvecs.get_int(chuncks_offset + chunck_idx * subchunck_table_offset + subchunck_idx * subchunck_rank_bits, subchunck_rank_bits);

      int rel_pos = i - subchunck_offset - offset;

      uint64_t count = bitvec->get_int(subchunck_offset + offset, rel_pos);
      int popcount = sdsl::bits::cnt(count);
      return chunck_rank + subchunck_rank + popcount;
   }

   uint64_t overhead(){
      uint64_t numbits = subchunk_bitvecs.size();
      return numbits;
   }

   void print(){
      std::cout << "bitvector size " << bitvec->size() << std::endl;
        for(int i = 0; i < bitvec->size(); i++){
          std::cout << "i " << i << " bit vector " << bitvec->get_int(i, 1) <<std::endl;
        }
   }
   uint64_t size(){
      return bitvec->size();
   }

   void save(std::string& fname){
      sdsl::store_to_file(subchunk_bitvecs, fname);
   }

};

class my_select_vector{
   //  rv;
   my_rank_vector* rv;
   public:
   my_select_vector(my_rank_vector* r){
      rv = r;
   }

   int binarySearch(int target, int l, int r){

    while(l < r){
        uint64_t m = (l + r) / 2;
        uint64_t rank_m = rv->rank1(m);        

      //   std::cout << "l " << l << " m " << m << " r " << r <<std::endl;
        if (rank_m < target){  // query > as[m]
            l = m+1;
        } else if (rank_m > target){  // query < as[m]
            r = m;
        } else if (l != r){
            r = m;
        }
        else
            return m;

    }
    return l;
}

   uint64_t select1(uint64_t i){
      return binarySearch(i, 0, rv->size());
   }

   uint64_t overhead(){
      return rv->overhead();
   }
};


class sparse_array{
   uint64_t array_size;
   uint64_t count;
   // uint64_t prev_pos = std::numeric_limits<uint64_t>::max();
   int prev_pos = -1;
   sdsl::int_vector<1> bit_vec;
   std::vector<std::string> strings;
   my_rank_vector* rv;
   my_select_vector* sv;
   bool finalized = false;
   public:
   sparse_array(uint64_t size){
      array_size = size;
      bit_vec = sdsl::int_vector<1>(size, 0, 1);
   }

   void append(std::string elem, uint64_t pos){
      long signed_pos = pos;
      assert(signed_pos > prev_pos);
      assert(pos < array_size);
      count += 1;
      prev_pos = pos;

      bit_vec[pos] = 1;
      strings.push_back(elem);
   }

   void finalize(){
      rv = new my_rank_vector(&bit_vec);
      sv = new my_select_vector(rv);
      finalized = true;
   }

   bool get_at_rank(uint64_t r, std::string& elem){
      if(r > count && r == 0){
         return false;
      }
      elem = strings[r-1];
      return true;
   }

   bool get_at_index(uint64_t r, std::string& elem){
      assert(finalized);
      if(r >= array_size || bit_vec[r] == 0)
         return false;
      int index = rv->rank1(r);
      elem = strings[index];
      return true;
   }

   uint64_t get_index_of(uint64_t r){
      assert(finalized);
      if(r > count || r == 0){
         return std::numeric_limits<uint64_t>::max();
      }
      return sv->select1(r)-1;
   }

   uint64_t size(){
      return array_size;
   }

   uint64_t num_elem(){
      return count;
   }

   void save(std::string& bitvector_fname, std::string& string_elts_fname){
      save_bitvec(bit_vec, bitvector_fname);

      std::ofstream os(string_elts_fname, std::ios::binary);
      std::stringstream out_buffer;

      out_buffer << strings.size() <<  "\n";

      for (size_t i = 0; i < strings.size(); ++i) {
         out_buffer << strings[i] <<  "\n";
      }
      out_buffer << "\0";
      cereal::BinaryOutputArchive oarchive(os); // Create an output archive
      oarchive(out_buffer.str());
   }

   void load(std::string& bitvector_fname, std::string& string_elts_fname){

   load_bitvec(bit_vec, bitvector_fname);
   array_size = bit_vec.size();
   std::string inputBuffer;
   std::ifstream os(string_elts_fname, std::ios::binary);
   std::string line;
    cereal::BinaryInputArchive iarchive(os); // Create an output archive
    try {
        // std::cout << "before opening" << std::endl;
        iarchive(inputBuffer);
        // std::cout << "out buffer " << dto2 << std::endl;
    }
    catch (std::runtime_error e) {
        e.what();
    }


   std::istringstream istr (inputBuffer);

    getline(istr, line);
    count = std::stoi(line);

    for(int i = 0; i < count; i++){
      getline(istr, line);
      strings.push_back(line);
    }
   finalize();
    }

};



void test_rank_support(){
   int n = 10000;
   int logn = log2(n);
   sdsl::int_vector<1> bit_vec_og = sdsl::int_vector<1>(n, 0, 1); // TODO switch to compact representation

   sdsl::util::set_random_bits(bit_vec_og);
   my_rank_vector r = my_rank_vector(&bit_vec_og);

   // std::cout << bit_vec_og << std::endl;
   
   for(int i = 0; i < n; i++){
      int rank_naive = 0;
      for(int j = 0; j < i; j++){
         rank_naive += bit_vec_og[j];
      }
            // std::cout <<  "rank of  " << i << ": " << r.rank1(i) << " naive rank " << rank_naive << '\n';
      assert(rank_naive == r.rank1(i));
   }

   // my_select_vector s = my_select_vector(&r);
   // int s14 = s.select1(19);
   // std::cout <<  "overhead  " << r.overhead() << '\n';
   // std::cout <<  "select of  " << 19 << ": " << s14 << '\n';
   // std::cout << " rank " << s14 << " " <<r.rank1(s14) << std::endl;
}

void test_select_support(){
   int n = 10;
   int logn = log2(n);
   sdsl::int_vector<1> bit_vec_og = sdsl::int_vector<1>(n, 0, 1); // TODO switch to compact representation
   bit_vec_og[n-1] = 1;
   // sdsl::util::set_random_bits(bit_vec_og);
   my_rank_vector r = my_rank_vector(&bit_vec_og);

   my_select_vector s = my_select_vector(&r);
   int s1 = s.select1(1);
   std::cout <<  "overhead  " << r.overhead() << '\n';
   std::cout <<  "select of  " << 1 << ": " << s1 << '\n';
}

void test_saving_rank(){
   int n = 10000;
   int logn = log2(n);
   sdsl::int_vector<1> bit_vec_og = sdsl::int_vector<1>(n, 0, 1); // TODO switch to compact representation
   // for(int i = 0; i<n; i+=4){
   //    bit_vec_og[i] = 1;
   // }
   sdsl::util::set_random_bits(bit_vec_og);
   my_rank_vector r = my_rank_vector(&bit_vec_og);

   std::string filename = "saved_rank.bin";
   std::string bitvec_filename = "bitvec.bin";

   r.save(filename);
   save_bitvec(bit_vec_og, bitvec_filename);
}

void test_loading_rank(){
   std::string filename = "saved_rank.bin";
   std::string bitvec_filename = "bitvec.bin";

   sdsl::int_vector<1> bit_vec_og;
   load_bitvec(bit_vec_og, bitvec_filename);

   my_rank_vector r = my_rank_vector(&bit_vec_og, filename);

   for(int i = 0; i < bit_vec_og.size(); i++){
      int rank_naive = 0;
      for(int j = 0; j < i; j++){
         rank_naive += bit_vec_og[j];
      }
            // std::cout <<  "rank of  " << i << ": " << r.rank1(i) << " naive rank " << rank_naive << '\n';
      assert(rank_naive == r.rank1(i));
   }

}

void print_overheads(){
   int ns[] = {10, 100, 500, 1000, 10000, 50000, 100000, 500000};
    int num_ns = sizeof(ns) / sizeof(ns[0]);

    std::cout << "--------------------------" << std::endl;

   for(int i = 0; i < num_ns; i++){
      int n = ns[i];
      sdsl::int_vector<1> bit_vec_og = sdsl::int_vector<1>(n, 0, 1);
      my_rank_vector r = my_rank_vector(&bit_vec_og);
      std::cout << "size " << n << " overhead " << r.overhead() << std::endl;
   }


   
    std::cout << "--------------------------" << std::endl;
}


void print_rank_times(){
   int ns[] = {10, 100, 500, 1000, 10000, 50000, 100000, 500000};
    int num_ns = sizeof(ns) / sizeof(ns[0]);

    std::cout << "--------------------------" << std::endl;
    
    std::cout << "size vs rank time" << std::endl;
   for(int i = 0; i < num_ns; i++){
      int n = ns[i];
      sdsl::int_vector<1> bit_vec_og = sdsl::int_vector<1>(n, 0, 1);
      my_rank_vector r = my_rank_vector(&bit_vec_og);

      auto start = high_resolution_clock::now();

      for(int k = 0; k < 100000; k++){
         float prob = float(std::rand()) / INT_MAX;
         int rand_rank = prob * n ;
         r.rank1(rand_rank);
      }

      auto end = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(end - start);
      std::cout << n << " " <<  duration.count() <<std::endl;
   }
    std::cout << "--------------------------" << std::endl;
}

void print_select_times(){
   int ns[] = {10, 100, 500, 1000, 10000, 50000, 100000, 500000};
    int num_ns = sizeof(ns) / sizeof(ns[0]);

    std::cout << "--------------------------" << std::endl;
    
    std::cout << "size vs select time" << std::endl;
   for(int i = 0; i < num_ns; i++){
      int n = ns[i];
      sdsl::int_vector<1> bit_vec_og = sdsl::int_vector<1>(n, 0, 1);
      my_rank_vector r = my_rank_vector(&bit_vec_og);
      my_select_vector s = my_select_vector(&r);

      auto start = high_resolution_clock::now();

      for(int k = 0; k < 100000; k++){
         float prob = float(std::rand()) / INT_MAX;
         int rand_rank = prob * n * 0.4;
         s.select1(rand_rank);
      }

      auto end = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(end - start);
      std::cout << n << " " <<  duration.count() <<std::endl;
   }
    std::cout << "--------------------------" << std::endl;
}

void test_sparse_array(){
   sparse_array sp = sparse_array(200);

   sp.append("a", 0);
   sp.append("b", 10);
   sp.append("c", 11);
   sp.append("d", 99);
   sp.append("", 150);
   sp.finalize();

   std::string elem;
   bool flag;
   flag = sp.get_at_rank(3, elem);
   assert(elem=="c");

   flag = sp.get_at_rank(1, elem);
   assert(elem=="a");

   flag = sp.get_at_index(10, elem);

   assert(elem=="b");
   assert(flag);

   flag = sp.get_at_index(15, elem);
   assert(!flag);


   int index_1= sp.get_index_of(1);
   int index_2 = sp.get_index_of(2);
   assert(index_1 == 0);
   assert(index_2 == 10);

   std::string filea = "a.bin";
   std::string fileb = "b.bin";


   sparse_array sp2 = sparse_array(1);
    sp.save(filea, fileb);
    sp2.load(filea, fileb);
}

void test_loaded_sparse_array(){
   std::string filea = "a.bin";
   std::string fileb = "b.bin";


   sparse_array sp = sparse_array(1);
    sp.load(filea, fileb);

   std::string elem;
   bool flag;
   flag = sp.get_at_rank(3, elem);
   assert(elem=="c");

   flag = sp.get_at_rank(1, elem);
   assert(elem=="a");

   flag = sp.get_at_index(10, elem);

   assert(elem=="b");
   assert(flag);

   flag = sp.get_at_index(15, elem);
   assert(!flag);


   int index_1= sp.get_index_of(1);
   int index_2 = sp.get_index_of(2);
   assert(index_1 == 0);
   assert(index_2 == 10);
}

void generate_different_sparsities(){
   float sparsities[] = {0.01, 0.05, 0.1};
   for(int j = 0; j < 3; j++){
      int n = 10000;
      sparse_array sp = sparse_array(n);
      int counts = 0;
   for(int i = 0; i < n; i ++){
      float prob = float(std::rand()) / INT_MAX;
      if(prob < sparsities[j]){
         sp.append("a", i);
         counts+=1;
      }
   }
   sp.finalize();
      std::cout << " counts " << counts << std::endl;

      auto start = high_resolution_clock::now();
      std::string elet;

      for(int k = 0; k < 100000; k++){
         float prob = float(std::rand()) / INT_MAX;
         int rand_rank = prob * sparsities[j] * n * 0.8;
         sp.get_at_rank(rand_rank, elet);
      }

      std::cout << "sparsity " << sparsities[j] << std::endl;
      auto end = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(end - start);
      std::cout << "get at rank time " <<  duration.count() <<std::endl;

      start = high_resolution_clock::now();

      for(int k = 0; k < 100000; k++){
         float prob = float(std::rand()) / INT_MAX;
         int rand_rank = prob * n ;
         sp.get_at_index(rand_rank, elet);
      }

      end = high_resolution_clock::now();
      duration = duration_cast<microseconds>(end - start);
      std::cout << "get at index time " <<  duration.count() <<std::endl;

   }
}

void generate_different_sizes(){
   float ns[] = {100, 10000, 100000, 1000000};
   for(int j = 0; j < 4; j++){
      int n = ns[j];
      sparse_array sp = sparse_array(n);
      int counts = 0;
   for(int i = 0; i < n; i ++){
      float prob = float(std::rand()) / INT_MAX;
      if(prob < 0.05){
         sp.append("a", i);
         counts+=1;
      }
   }
   sp.finalize();

      auto start = high_resolution_clock::now();
      std::string elet;

      for(int k = 0; k < 100000; k++){
         float prob = float(std::rand()) / INT_MAX;
         int rand_rank = prob * 0.05 * n * 0.8;
         sp.get_at_rank(rand_rank, elet);
      }

      std::cout << "size " << n << std::endl;
      auto end = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(end - start);
      std::cout << "get at rank time " <<  duration.count() <<std::endl;

      start = high_resolution_clock::now();

      for(int k = 0; k < 100000; k++){
         float prob = float(std::rand()) / INT_MAX;
         int rand_rank = prob * n ;
         sp.get_at_index(rand_rank, elet);
      }

      end = high_resolution_clock::now();
      duration = duration_cast<microseconds>(end - start);
      std::cout << "get at index time " <<  duration.count() <<std::endl;

   }
}


int main()
{
   test_rank_support(); 
   test_saving_rank();
   test_loading_rank();
   print_overheads();
   test_select_support();
   test_sparse_array();
   test_loaded_sparse_array();
   generate_different_sparsities();
   generate_different_sizes();
   print_rank_times();
   print_select_times();

   return 0;
}
