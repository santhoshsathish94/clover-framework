#include "k3_tok.h"

int main(int argc,char **argv)
{
    if(argc!=3)return 2;
    Tok tokenizer;k3_tok_load(&tokenizer,argv[1]);
    int ids[256];int count=tok_encode(&tokenizer,argv[2],(int)strlen(argv[2]),ids,256);
    if(count<=0||count>=256)return 1;
    for(int index=0;index<count;index++)printf("%s%d",index?",":"",ids[index]);
    putchar('\n');return 0;
}