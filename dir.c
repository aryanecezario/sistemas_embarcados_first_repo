/* 
 * Exibe arquivos e diretórios de um local 
 * apontado pelo usuário através de parâmetro 
 * de entrada.
 */

#include <stdio.h>
#include <dirent.h>


int main(int argc, char** argv){

    DIR *folder;
    struct dirent *entry;
    int files = 0;

    /* Se o comando tiver menos de dois argumentos
       exiba uma mensagem para o usuário informando 
       os parâmetros necessários. */
    if(argc < 2){
        printf("Uso:\n\t%s <dir>\n", argv[0]);
        return 0;
    }

    /* Abre diretório informado pelo usuário */
    folder = opendir(argv[1]);

    if(folder == NULL)
    {
        perror("Não foi possível abrir o diretório;\n");
        return 1;
    }

    while( (entry=readdir(folder)) )
    {
        files++;

        printf("File %3d: ", files);
        
        /* Diretórios são exibidos entre [] */
        if(entry->d_type==DT_DIR){
            printf("[%s]\n", entry->d_name);
        }else{
            printf("%s\n", entry->d_name);
        }
    }

    closedir(folder);

    return(0);
}