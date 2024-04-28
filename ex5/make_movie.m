clear all
close all
clc

load('Results_10000Iterations')

create_movie = 1; 
if create_movie
    movie_name = 'Learning';  % file name
    vidfile = VideoWriter(movie_name,'MPEG-4');
    vidfile.FrameRate = 10;      % change this number to slow down or speed up the movie
    open(vidfile);
    fig = figure;
    set(fig,'color','w');
end

for ii = 1:NT
    Xii = XX1(:,ii);
    Xii = reshape(Xii,2,[]);
    plot(Xii(1,1:I/2), Xii(2,1:I/2), 'bo')
    hold on
    plot(Xii(1,I/2+(1:I/2)), Xii(2,I/2+(1:I/2)), 'ro')
    plot(0, 1,'bx')
    plot(2, 1,'rx')
    hold off
    xlabel 'x_1'
    ylabel 'x_2'
    axis([-2, 4, -4, 4])
    title(['t = ', num2str(time(ii)), ' [s]'])
    if create_movie
        frame = getframe(fig);
        writeVideo(vidfile, frame);
    end
    pause(0.1);
end

if create_movie
    close(vidfile);
end