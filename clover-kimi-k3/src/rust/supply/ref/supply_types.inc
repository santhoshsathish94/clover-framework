struct ExpertStats { unsigned long needed = 0, hit_card = 0, hit_host = 0, from_drive = 0, written_back = 0; double t_read = 0, t_copy = 0, t_ensure = 0; };
enum { MAX_READERS = 32 };
